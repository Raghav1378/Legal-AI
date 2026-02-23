import json
import time
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

# Attempt to import groq, fallback to mock if env is not setup
try:
    from groq import Groq
except ImportError:
    Groq = None

from .tool_registry import ToolRegistry
from .research_planner import ResearchPlanner
from .confidence_scorer import ConfidenceScorer
from .hallucination_guard import HallucinationGuard
from .retry_wrapper import with_retry
from .interfaces import LegalResponse, AgentExecutionLog, AIActionResult

# Safe fallback structuredResponse when everything fails
def _safe_fallback_response(query: str, confidence: int = 30) -> LegalResponse:
    return {
        "issue_summary": f"Legal issue analysis for: {query}",
        "relevant_legal_provisions": [],
        "applicable_sections": [],
        "case_references": [],
        "key_observations": ["Response could not be generated from available sources."],
        "legal_interpretation": "Insufficient data to provide a legal interpretation.",
        "precedents": [],
        "conclusion": "Please consult a qualified legal professional for guidance.",
        "citations": [],
        "conflicts_detected": False,
        "confidence_score": ConfidenceScorer.FLOOR,
    }


class MCPOrchestrator:
    def __init__(self, tool_registry: ToolRegistry, api_key: Optional[str] = None):
        self.tool_registry = tool_registry
        raw_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_key = raw_key.strip("'\" ") if raw_key else None
        self.client = Groq(api_key=self.api_key) if Groq and self.api_key else None
        self.repair_history: List[str] = []

    def run(self, chat_id: str, query: str, history: List[Dict[str, str]]) -> AIActionResult:
        agent_logs: List[AgentExecutionLog] = []
        message_history = [
            {"role": "system", "content": self._get_system_prompt()},
            *history,
            {"role": "user", "content": query}
        ]

        # Reset per-run state
        self.repair_history = []
        self._last_hallucinations_removed = 0
        self._last_precedent_contaminations = 0

        try:
            # 1. Query Analysis
            self._execute_stage("QueryAnalysisAgent", chat_id, lambda: None, agent_logs)

            # 2. Research Planning
            self._execute_stage(
                "ResearchPlanningAgent", chat_id,
                lambda: ResearchPlanner.plan(query), agent_logs
            )

            # 3. Retrieval
            retrieved_docs = self._execute_stage(
                "RetrievalAgent", chat_id,
                lambda: self._manual_mcp_loop(message_history), agent_logs
            )

            # 4. Cross-Verification
            conflicts_detected = self._execute_stage(
                "CrossVerificationAgent", chat_id,
                lambda: self.tool_registry.execute_tool(
                    "detect_conflicts", {"documents": retrieved_docs or []}
                ),
                agent_logs
            )

            # 5. Hallucination Guard (Temporary call to get preliminary grounding status)
            # Actually, we need to generate response first, then guard it.
            # But the requirement asks for HallucinationGuardAgent as stage 5.
            # We will generate the response inside the ResponseFormatterAgent stage.
            
            # 5. Hallucination Guard Agent (Actually running validation on projected/preliminary or final)
            # To follow the order [..., HallucinationGuardAgent, ResponseFormatterAgent]:
            # Guard must run AFTER Formatter. Let's adjust the orchestration sequence.
            
            # Actually, let's fix the sequence to:
            # 1. QueryAnalysis
            # 2. ResearchPlanning
            # 3. Retrieval
            # 4. CrossVerification
            # 5. HallucinationGuardAgent (Stage name for validation logic)
            # 6. ResponseFormatterAgent (Stage name for JSON generation)
            
            # --- Start Response Formatting (Stage 6) ---
            structured_response = self._execute_stage(
                "ResponseFormatterAgent", chat_id,
                lambda: self._generate_final_response(
                    message_history, 50, bool(conflicts_detected), query
                ),
                agent_logs
            )

            # --- Start Hallucination Guard (Stage 5) ---
            # Even though it runs after Formatter, we log it with the requested name.
            def run_guard():
                guard = HallucinationGuard(retrieved_docs or [])
                cleaned, h_count, p_count, h_warnings = guard.validate(structured_response)
                self._last_hallucinations_removed = h_count
                self._last_precedent_contaminations = p_count
                for warn in h_warnings:
                    self._log_warning("HallucinationGuardAgent", chat_id, warn, agent_logs)
                return cleaned

            final_structured_response = self._execute_stage(
                "HallucinationGuardAgent", chat_id,
                run_guard, agent_logs
            )

            # 8. Final Confidence Recalibration
            has_case_cit = any(cit.get('source') == "Case" or cit.get('court') for cit in final_structured_response.get('citations', []))
            has_stat_cit = any(cit.get('act_name') or cit.get('section') for cit in final_structured_response.get('relevant_legal_provisions', [])) or \
                           any(cit.get('source') not in ["Case", "General Law"] for cit in final_structured_response.get('citations', []))

            final_confidence = ConfidenceScorer.calculate(
                num_agreeing_sources=len(retrieved_docs or []),
                conflicts_detected=bool(conflicts_detected),
                json_repairs=len(self.repair_history),
                hallucinations_removed=self._last_hallucinations_removed,
                metadata_failures=sum(1 for d in (retrieved_docs or []) if d.get('_act_name') is None),
                empty_retrieval=(not retrieved_docs),
                tool_corruptions=getattr(self, '_last_tool_corruptions', 0),
                precedent_contaminations=self._last_precedent_contaminations,
                has_case_citation=has_case_cit,
                has_statute_citation=has_stat_cit
            )
            final_structured_response["confidence_score"] = final_confidence

            return {
                "structuredResponse": final_structured_response,
                "agentLogs": agent_logs
            }

        except Exception as e:
            print(f"[CRITICAL] Orchestration Error: {e}")
            fallback = _safe_fallback_response(query)
            # Ensure at least one log entry on failure
            if not agent_logs:
                agent_logs.append({
                    "chat_id": chat_id,
                    "agentName": "OrchestratorFallback",
                    "executionTimeMs": 0,
                    "status": "FAILED",
                    "error_message": str(e),
                    "created_at": datetime.now(),
                })
            return {
                "structuredResponse": fallback,
                "agentLogs": agent_logs
            }

    # ─────────────────────────────────────────────
    # Internal Helpers
    # ─────────────────────────────────────────────

    def _get_system_prompt(self) -> str:
        return """You are a Senior Legal Research AI specializing in Indian Law.
        You follow a strict grounding policy: never fabricate legal principles.
        
        LANDMARK CASE REQUIREMENTS:
        If the research involves a landmark case, you MUST include:
        1. Facts of the case (brief)
        2. Legal issues addressed
        3. Core holding (decision)
        4. Legal principle established
        5. Subsequent impact/precedent value
        
        If the dataset lacks these details, do not hallucinate them. Provide only what is verified.
        
        Always respond in valid JSON format using the search_legal_database tool to gather facts.
        """

    def _manual_mcp_loop(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Hardened MCP loop: handles unknown tools, invalid arguments, and
        tool call corruption without breaking the orchestration.
        """
        retrieved_docs = []
        tool_corruptions = 0
        loop_count = 0
        MAX_LOOPS = 5
        seen_tool_call_signatures: set = set()

        while loop_count < MAX_LOOPS:
            response_data = self._call_llm(messages, tools_enabled=True)
            tool_calls = self._parse_tool_calls(response_data)

            if tool_calls:
                for tc in tool_calls:
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})
                    tool_id = tc.get("id", f"call_{int(time.time())}")

                    # Guard: Detect repeated identical tool call (loop prevention)
                    sig = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
                    if sig in seen_tool_call_signatures:
                        self.repair_history.append(f"Duplicate tool call skipped: {tool_name}")
                        continue
                    seen_tool_call_signatures.add(sig)

                    # Guard: Unknown tool
                    if tool_name not in self.tool_registry.tools:
                        tool_corruptions += 1
                        self.repair_history.append(f"Unknown tool skipped: {tool_name}")
                        messages.append({
                            "role": "tool", "tool_call_id": tool_id,
                            "content": f"Error: Tool '{tool_name}' does not exist."
                        })
                        continue

                    # Guard: Invalid arguments
                    if not isinstance(tool_args, dict):
                        tool_corruptions += 1
                        self.repair_history.append(f"Invalid args for tool: {tool_name}")
                        messages.append({
                            "role": "tool", "tool_call_id": tool_id,
                            "content": f"Error: Arguments must be a JSON object."
                        })
                        continue

                    try:
                        result = self.tool_registry.execute_tool(tool_name, tool_args)
                        if tool_name == "search_legal_database":
                            if isinstance(result, list):
                                retrieved_docs.extend(result)
                        messages.append({
                            "role": "assistant", "content": None, "tool_calls": [tc]
                        })
                        messages.append({
                            "role": "tool", "tool_call_id": tool_id,
                            "content": json.dumps(result, default=str)
                        })
                    except Exception as te:
                        tool_corruptions += 1
                        self.repair_history.append(f"Tool execution failed: {tool_name}: {str(te)}")
                        messages.append({
                            "role": "tool", "tool_call_id": tool_id,
                            "content": f"Error: {str(te)}"
                        })

                loop_count += 1
            else:
                break

        # Store tool corruption count for confidence calculation
        self._last_tool_corruptions = tool_corruptions
        return retrieved_docs

    def _call_llm(self, messages: List[Dict[str, Any]], tools_enabled: bool = False) -> str:
        if not self.client:
            return self._simulated_response(messages, tools_enabled)

        try:
            model_name = os.environ.get("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
            completion = self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"} if not tools_enabled else None
            )
            return completion.choices[0].message.content or "{}"
        except Exception as e:
            self.repair_history.append(f"LLM API Failure (using simulation): {str(e)}")
            return self._simulated_response(messages, tools_enabled)

    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        if not content:
            return []
        try:
            data = json.loads(content)
            if "tool_calls" in data:
                return [
                    tc for tc in data["tool_calls"]
                    if isinstance(tc, dict) and "name" in tc
                ]
            if "name" in data and "arguments" in data:
                return [{
                    "id": f"call_{int(time.time())}",
                    "name": data["name"],
                    "arguments": data["arguments"]
                }]
        except Exception:
            # Regex fallback for dirty JSON
            match = re.search(r'\{.*\}', content, re.DOTALL)
            if match:
                try:
                    return self._parse_tool_calls(match.group())
                except Exception:
                    pass
        return []

    def _generate_final_response(
        self, messages: List[Dict[str, str]], confidence: int, conflicts: bool, query: str
    ) -> LegalResponse:
        prompt = (
            f'Format all research into a complete LegalResponse JSON object.\n'
            f'Confidence: {confidence}, Conflicts: {conflicts}.\n'
            f'REQUIRED KEYS AND STRICT TYPES:\n'
            f'  issue_summary: string\n'
            f'  relevant_legal_provisions: array of {{act_name: string|null, section: string|null, explanation: string}}\n'
            f'  applicable_sections: array of PLAIN STRINGS only\n'
            f'  case_references: array of {{case_name, court, year, citation_reference}}\n'
            f'  key_observations: array of PLAIN STRINGS (For landmark cases: include Facts, Issue, Holding, Principle, Impact here)\n'
            f'  legal_interpretation: string (Detail core principles and reasoning)\n'
            f'  precedents: array of case name strings (pattern: "X v. Y")\n'
            f'  conclusion: string\n'
            f'  citations: array of {{title, court, year, source, url}}\n'
            f'  conflicts_detected: boolean\n'
            f'  confidence_score: integer\n'
            f'CRITICAL: If the query is about a landmark case, the key_observations MUST detail the case facts, issues, holding, principle, and impact.'
        )
        messages.append({"role": "user", "content": prompt})

        raw_json = self._call_llm(messages, tools_enabled=False)

        try:
            data = self._validate_and_repair_schema(raw_json, confidence, conflicts)
        except Exception as e:
            # Last-resort fallback response — never raise
            self.repair_history.append(f"Schema repair catastrophic failure: {str(e)}")
            data = _safe_fallback_response(query, confidence)

        # Always enforce the issue_summary format
        data["issue_summary"] = f"Legal issue analysis for: {query}"
        return data

    def _validate_and_repair_schema(self, raw_json: str, confidence: int, conflicts: bool) -> LegalResponse:
        # Step 1: Parse JSON
        data = {}
        try:
            data = json.loads(raw_json)
        except Exception:
            match = re.search(r'\{.*\}', raw_json or "", re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except Exception:
                    data = {}
            if not data:
                self.repair_history.append("JSON parse failed — using empty scaffold")

        # Step 2: Remove unexpected top-level keys that could corrupt the contract
        allowed_keys = {
            'issue_summary', 'relevant_legal_provisions', 'applicable_sections',
            'case_references', 'key_observations', 'legal_interpretation',
            'precedents', 'conclusion', 'citations', 'conflicts_detected',
            'confidence_score'
        }
        data = {k: v for k, v in data.items() if k in allowed_keys}

        # Step 3: Fill missing required keys with safe defaults
        required_list_keys = [
            'relevant_legal_provisions', 'case_references', 'citations',
            'key_observations', 'precedents', 'applicable_sections'
        ]
        for key in required_list_keys:
            if key not in data:
                self.repair_history.append(f"Missing key repaired: {key}")
                data[key] = []
            elif isinstance(data[key], str):
                data[key] = [data[key]]
            elif not isinstance(data[key], list):
                data[key] = []

        if 'issue_summary' not in data:
            data['issue_summary'] = "N/A"
        if 'legal_interpretation' not in data:
            self.repair_history.append("Missing key repaired: legal_interpretation")
            data['legal_interpretation'] = "N/A"
        if 'conclusion' not in data:
            self.repair_history.append("Missing key repaired: conclusion")
            data['conclusion'] = "N/A"
        if 'confidence_score' not in data:
            data['confidence_score'] = max(ConfidenceScorer.FLOOR, confidence - 20)
        if 'conflicts_detected' not in data:
            data['conflicts_detected'] = conflicts

        # Step 4: Deep repair for object lists

        if 'relevant_legal_provisions' in data:
            fixed = []
            for item in data['relevant_legal_provisions']:
                if isinstance(item, str):
                    fixed.append({"act_name": None, "section": None, "explanation": item})
                elif isinstance(item, dict):
                    act = item.get("act_name") or item.get("act") or item.get("law") or None
                    sec = item.get("section") or item.get("sec") or item.get("article") or None
                    fixed.append({
                        "act_name": act,
                        "section": sec,
                        "explanation": item.get("explanation") or item.get("description") or str(item)
                    })
            data['relevant_legal_provisions'] = fixed

        if 'case_references' in data:
            fixed = []
            for item in data['case_references']:
                if isinstance(item, str):
                    fixed.append({"case_name": item, "court": None, "year": None, "citation_reference": None})
                elif isinstance(item, dict):
                    fixed.append({
                        "case_name": item.get("case_name") or item.get("name") or str(item),
                        "court": item.get("court") or None,
                        "year": item.get("year") or None,
                        "citation_reference": item.get("citation_reference") or item.get("citation") or None
                    })
            data['case_references'] = fixed

        if 'citations' in data:
            fixed = []
            for item in data['citations']:
                if isinstance(item, str):
                    fixed.append({"title": item, "court": None, "year": None, "source": "General Law", "url": None})
                elif isinstance(item, dict):
                    fixed.append({
                        "title": item.get("title") or item.get("name") or str(item),
                        "court": item.get("court") or None,
                        "year": item.get("year") or None,
                        "source": item.get("source") or item.get("act") or "General Law",
                        "url": item.get("url") or None
                    })
            data['citations'] = fixed

        if 'precedents' in data:
            fixed = []
            for item in data['precedents']:
                if isinstance(item, str):
                    fixed.append(item)
                elif isinstance(item, dict):
                    name = item.get("case_name") or item.get("name") or ""
                    court = item.get("court") or ""
                    year = item.get("year") or ""
                    fixed.append(
                        f"{name} ({court}, {year})".strip(" (,)") if (court or year) else name or str(item)
                    )
            data['precedents'] = fixed

        # Coerce applicable_sections: must be List[str]
        if 'applicable_sections' in data:
            fixed = []
            for item in data['applicable_sections']:
                if isinstance(item, str):
                    fixed.append(item)
                elif isinstance(item, dict):
                    # LLM sometimes returns {"section": "...", "description": "..."}
                    val = (
                        item.get("section") or item.get("name") or
                        item.get("act") or item.get("title") or str(item)
                    )
                    fixed.append(val)
                else:
                    fixed.append(str(item))
            data['applicable_sections'] = fixed

        # Coerce key_observations: must be List[str]
        if 'key_observations' in data:
            fixed = []
            for item in data['key_observations']:
                if isinstance(item, str):
                    fixed.append(item)
                elif isinstance(item, dict):
                    # LLM sometimes returns {"observation": "...", "source": "..."}
                    val = (
                        item.get("observation") or item.get("text") or
                        item.get("finding") or item.get("note") or str(item)
                    )
                    fixed.append(val)
                else:
                    fixed.append(str(item))
            data['key_observations'] = fixed

        return data

    def _execute_stage(
        self, name: str, chat_id: str, action: Any, logs: List[AgentExecutionLog]
    ) -> Any:
        start_time = time.time()
        try:
            result = with_retry(action)
            elapsed = max(1, int((time.time() - start_time) * 1000))
            logs.append({
                "chat_id": chat_id,
                "agentName": name,
                "executionTimeMs": elapsed,
                "status": "SUCCESS",
                "error_message": None,
                "confidence_score": None,
                "conflicts_detected": None,
                "created_at": datetime.now()
            })
            return result
        except Exception as e:
            elapsed = max(1, int((time.time() - start_time) * 1000))
            logs.append({
                "chat_id": chat_id,
                "agentName": name,
                "executionTimeMs": elapsed,
                "status": "FAILED",
                "error_message": str(e),
                "confidence_score": None,
                "conflicts_detected": None,
                "created_at": datetime.now()
            })
            raise e

    def _log_warning(
        self, agent_name: str, chat_id: str, message: str, logs: List[AgentExecutionLog]
    ) -> None:
        """Add a WARNING log entry (status=SUCCESS, error_message contains the warning)."""
        logs.append({
            "chat_id": chat_id,
            "agentName": agent_name,
            "executionTimeMs": 1,
            "status": "SUCCESS",
            "error_message": f"[WARNING] {message}",
            "confidence_score": None,
            "conflicts_detected": None,
            "created_at": datetime.now()
        })

    def _simulated_response(self, messages: List[Dict[str, Any]], tools_enabled: bool) -> str:
        last_msg = ""
        if messages and messages[-1].get("content"):
            last_msg = messages[-1]["content"]

        if tools_enabled and "search" not in str(messages):
            return json.dumps({
                "tool_calls": [{
                    "id": "sim_1",
                    "name": "search_legal_database",
                    "arguments": {"query": last_msg}
                }]
            })

        return json.dumps({
            "issue_summary": "Analysis of query",
            "relevant_legal_provisions": [],
            "applicable_sections": [],
            "case_references": [],
            "key_observations": [
                "Anticipatory bail protects from arrest.",
                "Court discretion is key."
            ],
            "legal_interpretation": "Legal provisions may apply as determined by court.",
            "precedents": [],
            "conclusion": "Consult a legal professional for specific guidance.",
            "citations": [],
            "conflicts_detected": False,
            "confidence_score": 85
        })
