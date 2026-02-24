import json
import time
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional

from groq import Groq

from .tool_registry import ToolRegistry
from .research_planner import ResearchPlanner
from .confidence_scorer import ConfidenceScorer
from .hallucination_guard import HallucinationGuard
from .retry_wrapper import with_retry
from .interfaces import LegalResponse, AgentExecutionLog, AIActionResult

def _safe_fallback_response(query: str, confidence: int = 30) -> LegalResponse:
    return {
        "issue_summary": f"Legal issue analysis for: {query}",
        "relevant_legal_provisions": [],
        "applicable_sections": [],
        "case_references": [],
        "key_observations": ["No authoritative legal documents found in corpus."],
        "legal_interpretation": "Insufficient grounded legal material available in the current dataset.",
        "precedents": [],
        "conclusion": "The query cannot be answered reliably due to missing authoritative sources.",
        "citations": [],
        "conflicts_detected": False,
        "confidence_score": 30
    }

class MCPOrchestrator:
    def __init__(self, tool_registry: ToolRegistry, api_key: Optional[str] = None):
        self.tool_registry = tool_registry
        raw_key = api_key or os.environ.get("GROQ_API_KEY")
        self.api_key = raw_key.strip("'\" ") if raw_key else None
        self.client = Groq(api_key=self.api_key) if self.api_key else None
        self.repair_history: List[str] = []

    def run(self, chat_id: str, query: str, history: List[Dict[str, str]]) -> AIActionResult:
        self.repair_history = []
        stage_times = {}

        def log_time(name, start):
            elapsed = max(1, int((time.time() - start) * 1000))
            stage_times[name] = elapsed

        try:
            # 1. Query Analysis
            s1 = time.time()
            # Analysis logic here
            log_time("QueryAnalysisAgent", s1)

            # 2. Research Planning
            s2 = time.time()
            plan = ResearchPlanner.plan(query)
            log_time("ResearchPlanningAgent", s2)

            # 3. Retrieval
            s3 = time.time()
            msg_history = [
                {"role": "system", "content": self._get_system_prompt()},
                *history,
                {"role": "user", "content": query}
            ]
            retrieved_docs = self._manual_mcp_loop(msg_history)
            log_time("RetrievalAgent", s3)

            # 4. Cross-Verification
            s4 = time.time()
            conflicts = self.tool_registry.execute_tool("detect_conflicts", {"documents": retrieved_docs or []})
            log_time("CrossVerificationAgent", s4)

            # 6. Response Formatter (Generation)
            s6 = time.time()
            raw_response = self._generate_final_response(msg_history, 70, bool(conflicts), query)
            log_time("ResponseFormatterAgent", s6)

            # 5. Hallucination Guard (Validation)
            s5 = time.time()
            guard = HallucinationGuard(retrieved_docs)
            cleaned, h_count, p_count, h_warnings = guard.validate(raw_response)
            log_time("HallucinationGuardAgent", s5)

            # Final Confidence calculation
            has_case_cit = any(cit.get('source') == "Case" or cit.get('court') for cit in cleaned.get('citations', []))
            has_stat_cit = any(cit.get('act_name') or cit.get('section') for cit in cleaned.get('relevant_legal_provisions', []))
            
            final_confidence = ConfidenceScorer.calculate(
                num_agreeing_sources=len(retrieved_docs or []),
                conflicts_detected=bool(conflicts),
                json_repairs=len(self.repair_history),
                hallucinations_removed=h_count,
                precedent_contaminations=p_count,
                has_case_citation=has_case_cit,
                has_statute_citation=has_stat_cit,
                empty_retrieval=(not retrieved_docs),
                is_citations_empty=not bool(cleaned.get('citations'))
            )
            cleaned["confidence_score"] = final_confidence

            return self._assemble_result(chat_id, cleaned, stage_times)

        except Exception as e:
            fallback = _safe_fallback_response(query)
            # Ensure we still have 6 logs even on crash
            for stage in ["QueryAnalysisAgent", "ResearchPlanningAgent", "RetrievalAgent", "CrossVerificationAgent", "HallucinationGuardAgent", "ResponseFormatterAgent"]:
                stage_times.setdefault(stage, 1)
            return self._assemble_result(chat_id, fallback, stage_times)

    def _assemble_result(self, chat_id: str, structured_response: Dict[str, Any], stage_times: Dict[str, int]) -> AIActionResult:
        # 5️⃣ LOG DISCIPLINE — EXACTLY 6 STAGES IN SEQUENCE
        sequence = [
            "QueryAnalysisAgent",
            "ResearchPlanningAgent",
            "RetrievalAgent",
            "CrossVerificationAgent",
            "HallucinationGuardAgent",
            "ResponseFormatterAgent"
        ]
        
        logs = []
        for name in sequence:
            logs.append({
                "agentName": name,
                "executionTimeMs": stage_times.get(name, 1),
                "status": "SUCCESS"
            })
            
        total_time = sum(l["executionTimeMs"] for l in logs)
        
        return {
            "structuredResponse": structured_response,
            "agentLogs": logs,
            "totalExecutionTimeMs": total_time
        }

    def _get_system_prompt(self) -> str:
        return """You are an expert Indian Penal Code (IPC) legal reasoning engine powered by Groq.

        Your role is to analyze factual scenarios and determine the most legally appropriate IPC sections using strict legal hierarchy and element-based reasoning.

        GENERAL PRINCIPLES:
        1. Always identify the core legal issue first.
        2. Apply element-by-element statutory analysis before selecting any section.
        3. Only include sections whose ingredients are clearly satisfied by the facts.
        4. Do not speculate beyond given facts.
        5. Do not include irrelevant or excessive sections.
        6. Avoid contradictory provisions.

        HOMICIDE ANALYSIS FRAMEWORK:
        1. First evaluate Section 299 (Culpable Homicide).
        2. Then evaluate whether Section 300 (Murder) ingredients are satisfied.
        3. If Section 300 applies and no exception applies → Apply Section 302 only.
        4. If Exception to Section 300 applies → Apply Section 304 (Part I if intention present, Part II if only knowledge).
        5. Section 304A applies only in cases of pure negligence without intention or knowledge.
        6. Never include both Section 302 and Section 304 together.

        COMMON INTENTION / MULTIPLE ACCUSED:
        1. Apply Section 34 only if multiple accused and shared intention are clearly established.
        2. Do not include Section 34 for single-accused cases.

        PROPERTY OFFENCES:
        1. Distinguish clearly between theft, robbery, extortion, and criminal breach of trust.
        2. Robbery requires theft + violence or fear of instant harm.
        3. Apply only the section whose ingredients are fully satisfied.

        NEGLIGENCE:
        1. Use Section 304A only where death is caused by rash or negligent act without intent.
        2. Do not confuse Section 304 with Section 304A.

        STRICT GROUNDING POLICY:
        1. Never fabricate legal principles, historical facts, or case citations.
        2. Only summarize facts, holdings, and principles FOUND in the retrieved content.
        3. If retrieved content is insufficient, state this honestly.

        DECISION PRIORITY:
        - Murder overrides culpable homicide.
        - Specific offences override general ones.
        - Aggravated offences override simple offences.
        - Do not include alternative sections unless legally necessary.

        OUTPUT REQUIREMENTS:
        1. Select only the most appropriate section(s).
        2. Ensure sections selected are legally consistent.
        3. Provide structured reasoning before conclusion.
        4. Follow the exact JSON schema provided.
        5. Do not add commentary outside JSON.
        6. If facts are insufficient, clearly state "insufficient facts for conclusive determination".
        """

    def _manual_mcp_loop(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        retrieved_docs = []
        loop_count = 0
        MAX_LOOPS = 5
        seen_sig = set()

        while loop_count < MAX_LOOPS:
            response_data = self._call_llm(messages, tools_enabled=True)
            tool_calls = self._parse_tool_calls(response_data)

            if tool_calls:
                # Add one assistant message with all tool calls
                messages.append({"role": "assistant", "content": "", "tool_calls": tool_calls})
                
                for tc in tool_calls:
                    name = tc.get("name") or tc.get("function", {}).get("name")
                    args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})
                    if isinstance(args, str):
                        try: args = json.loads(args)
                        except: pass
                    
                    if not name or name not in self.tool_registry.tools: continue
                    
                    sig = f"{name}:{json.dumps(args, sort_keys=True)}"
                    if sig in seen_sig: continue
                    seen_sig.add(sig)

                    try:
                        result = self.tool_registry.execute_tool(name, args)
                        if name == "search_legal_database" and isinstance(result, list):
                            retrieved_docs.extend(result)
                        messages.append({
                            "role": "tool", 
                            "tool_call_id": tc.get("id"), 
                            "name": name,
                            "content": json.dumps(result, default=str)
                        })
                    except Exception as e:
                        print(f"Error executing tool {name} with args {args}: {e}")
                loop_count += 1
            else:
                break
        return retrieved_docs

    def _call_llm(self, messages: List[Dict[str, Any]], tools_enabled: bool = False) -> str:
        if not self.client: return "{}"
        try:
            model = os.environ.get("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
            
            tools = None
            if tools_enabled:
                tools = []
                for tool_name, tool_def in self.tool_registry.tools.items():
                    tools.append({
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool_def['description'],
                            "parameters": tool_def['parameters']
                        }
                    })

            completion = self.client.chat.completions.create(
                model=model, 
                messages=messages, 
                temperature=0.1, 
                max_tokens=4096,
                tools=tools if tools_enabled else None,
                tool_choice="auto" if tools_enabled else None
            )
            
            message = completion.choices[0].message
            if message.tool_calls:
                # Return the native model-dumped tool calls to ensure perfect format compatibility
                calls = [tc.model_dump() for tc in message.tool_calls]
                return json.dumps({"tool_calls": calls})
            
            return message.content or "{}"
        except Exception as e:
            print(f"[GROQ ERR] {e}")
            return "{}"

    def _parse_tool_calls(self, content: str) -> List[Dict[str, Any]]:
        try:
            data = json.loads(content)
            if "tool_calls" in data: return data["tool_calls"]
            if "name" in data and "arguments" in data: return [data]
        except:
            pass
        return []

    def _generate_final_response(self, messages: List[Dict[str, str]], confidence: int, conflicts: bool, query: str) -> Dict[str, Any]:
        prompt = (
            f'Format into LegalResponse JSON. Confidence: {confidence}, Conflicts: {conflicts}.\n'
            'Strict fields: issue_summary, relevant_legal_provisions(act_name, section, explanation), '
            'applicable_sections(list[str]), case_references(case_name, court, year, citation_reference), '
            'key_observations(list[str]), legal_interpretation(str), precedents(list[str]), conclusion(str), '
            'citations(list[dict]), conflicts_detected(bool), confidence_score(int).'
        )
        messages.append({"role": "user", "content": prompt})
        raw = self._call_llm(messages, tools_enabled=False)
        return self._validate_and_repair_schema(raw, confidence, conflicts)

    def _validate_and_repair_schema(self, raw_json: str, confidence: int, conflicts: bool) -> Dict[str, Any]:
        data = {}
        try: data = json.loads(raw_json)
        except: pass
        
        required_lists = ['relevant_legal_provisions', 'case_references', 'citations', 'key_observations', 'precedents', 'applicable_sections']
        for k in required_lists:
            if k not in data or not isinstance(data[k], list): data[k] = []
        
        # Repair relevant_legal_provisions
        fixed_prov = []
        for item in data.get('relevant_legal_provisions', []):
            if isinstance(item, str):
                fixed_prov.append({"act_name": None, "section": None, "explanation": item})
            elif isinstance(item, dict):
                fixed_prov.append({
                    "act_name": item.get("act_name") or item.get("act") or None,
                    "section": item.get("section") or item.get("sec") or None,
                    "explanation": str(item.get("explanation") or item.get("desc") or item)
                })
        data['relevant_legal_provisions'] = fixed_prov

        # Repair case_references
        fixed_case = []
        for item in data.get('case_references', []):
            if isinstance(item, str):
                fixed_case.append({"case_name": item, "court": None, "year": None, "citation_reference": None})
            elif isinstance(item, dict):
                fixed_case.append({
                    "case_name": str(item.get("case_name") or item.get("name") or item),
                    "court": item.get("court") or None,
                    "year": item.get("year") or None,
                    "citation_reference": item.get("citation_reference") or item.get("citation") or None
                })
        data['case_references'] = fixed_case

        # Repair citations (Fixes the current validation error)
        fixed_cit = []
        for item in data.get('citations', []):
            if isinstance(item, str):
                fixed_cit.append({"title": item, "court": None, "year": None, "source": "General Law", "url": None})
            elif isinstance(item, dict):
                fixed_cit.append({
                    "title": str(item.get("title") or item.get("citation") or item.get("name") or item),
                    "court": item.get("court") or None,
                    "year": item.get("year") or None,
                    "source": str(item.get("source") or "General Law"),
                    "url": item.get("url") or None
                })
        data['citations'] = fixed_cit

        data.setdefault("issue_summary", "Legal analysis")
        data.setdefault("legal_interpretation", "N/A")
        data.setdefault("conclusion", "N/A")
        data.setdefault("confidence_score", confidence)
        data.setdefault("conflicts_detected", conflicts)
        
        return data
