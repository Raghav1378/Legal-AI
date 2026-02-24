import os
import sys
import json
from datetime import datetime
from typing import Dict, Any

# Ensure we can import from src
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ai.mcp_orchestrator import MCPOrchestrator
from ai.tool_registry import ToolRegistry
from ai.retriever_service import RetrieverService

def _check(condition: bool, message: str):
    if not condition:
        print(f"FAILED: {message}")
        sys.exit(1)
    print(f"PASSED: {message}")

def _section(name: str):
    print(f"\n--- Testing: {name} ---")

def run_suite():
    retriever = RetrieverService()
    tools = ToolRegistry(retriever)
    orchestrator = MCPOrchestrator(tools)

    # 1. Empty Retrieval Test (Hybrid Mode)
    _section("Empty Retrieval Hybrid Mode")
    result = orchestrator.run("test_chat", "What is the punishment for a crime not in database?", [])
    
    _check("structuredResponse" in result, "structuredResponse key present")
    print(f"DEBUG: Score={result['structuredResponse']['confidence_score']}")
    _check(result["structuredResponse"]["confidence_score"] < 70, "Confidence score is penalized for empty retrieval")
    _check(len(result["structuredResponse"]["key_observations"]) > 0, "AI provided observations even with empty database")
    _check(len(result["agentLogs"]) == 6, "Exactly 6 agent logs")

    # 2. Log Discipline
    _section("Log Discipline")
    agent_names = [log["agentName"] for log in result["agentLogs"]]
    _check(len(set(agent_names)) == 6, "No duplicate agent names")
    
    expected_sequence = [
        "QueryAnalysisAgent", "ResearchPlanningAgent", "RetrievalAgent",
        "CrossVerificationAgent", "HallucinationGuardAgent", "ResponseFormatterAgent"
    ]
    _check(agent_names == expected_sequence, "Correct stage sequence")
    
    for log in result["agentLogs"]:
        _check(set(log.keys()) == {"agentName", "executionTimeMs", "status"}, f"Log entry {log['agentName']} has no extra keys")
        _check(log["executionTimeMs"] >= 1, f"Log entry {log['agentName']} execution time >= 1ms")

    _check("totalExecutionTimeMs" in result, "totalExecutionTimeMs present")
    _check(result["totalExecutionTimeMs"] == sum(l["executionTimeMs"] for l in result["agentLogs"]), "Total time matches sum of logs")

    # 3. Grounding & Section Normalization
    _section("Section Normalization & Precedent Purity")
    # Search for something that exists (Sibbia is in dataset.json)
    result_valid = orchestrator.run("test_chat", "What is Section 438 of CrPC in Gurbaksh Singh Sibbia?", [])
    
    structured = result_valid["structuredResponse"]
    
    # Check Section Formatting
    for prov in structured.get("relevant_legal_provisions", []):
        sec = prov.get("section")
        if sec:
            _check("Section" in sec or "Article" in sec, f"Section '{sec}' is normalized with prefix")
            _check(not sec.isdigit(), f"Section '{sec}' is not numeric-only")
            _check(sec.lower() != "unknown", "Section is not 'Unknown'")

    # Check Precedent Regex
    import re
    case_pattern = re.compile(r".+ v\. .+", re.IGNORECASE)
    for prec in structured.get("precedents", []):
        _check(bool(case_pattern.match(str(prec))), f"Precedent '{prec}' matches 'Party v. Party' pattern")

    # 4. Landmark Case Strict Mode
    _section("Landmark Case Strict Mode")
    result_landmark = orchestrator.run("test_chat", "Explain Gurbaksh Singh Sibbia v. State of Punjab", [])
    if "No authoritative legal documents found" in result_landmark["structuredResponse"]["key_observations"][0]:
        _check(result_landmark["structuredResponse"]["confidence_score"] == 30, "Landmark case fallback works when data missing")
    else:
        _check(len(result_landmark["structuredResponse"]["key_observations"]) > 0, "Landmark case has observations")

    print("\n✅ ALL PRODUCTION HARDENING TESTS PASSED")

if __name__ == "__main__":
    run_suite()
