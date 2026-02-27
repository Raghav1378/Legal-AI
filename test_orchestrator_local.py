import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from ai.ai_service import AIService
from ai.mcp_orchestrator import MCPOrchestrator
from ai.retriever_service import RetrieverService
from ai.tool_registry import ToolRegistry

def test_orchestrator():
    # Setup
    retriever = RetrieverService()
    tool_registry = ToolRegistry(retriever)
    orchestrator = MCPOrchestrator(tool_registry)
    
    query = "What are the essential elements of cheating under Section 415 of the IPC?"
    print(f"--- TESTING QUERY: {query} ---")
    
    result = orchestrator.run("test-chat", query, [])
    
    print("\n--- AGENT LOGS ---")
    for log in result.get("agentLogs", []):
        print(f"{log['agentName']}: {log['status']} ({log['executionTimeMs']}ms)")
        
    print("\n--- STRUCTURED RESPONSE ---")
    import json
    print(json.dumps(result["structuredResponse"], indent=2))

if __name__ == "__main__":
    test_orchestrator()
