import os
try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

from .memory_service import MemoryService
from .retriever_service import RetrieverService
from .tool_registry import ToolRegistry
from .mcp_orchestrator import MCPOrchestrator
from .interfaces import AIActionResult

class AIService:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.memory = MemoryService()
        self.retriever = RetrieverService()
        self.tool_registry = ToolRegistry(self.retriever)
        self.orchestrator = MCPOrchestrator(self.tool_registry, api_key=self.api_key)

    def process_legal_query(self, chat_id: str, query: str) -> AIActionResult:
        """
        Main entry point for the AI Engine.
        Returns strictly { structuredResponse, agentLogs, totalExecutionTimeMs }.
        Never raises exceptions to the caller.
        """
        history = self.memory.get_history(chat_id)
        result = self.orchestrator.run(chat_id, query, history)

        if self.orchestrator.repair_history:
            print(f"[WARN] Repairs for chat {chat_id}: {self.orchestrator.repair_history}")

        # Memory: store only factual conclusion to prevent memory corruption
        conclusion = result["structuredResponse"].get("conclusion", "")
        if conclusion and conclusion != "N/A":
            self.memory.add_message(chat_id, "user", query)
            self.memory.add_message(chat_id, "assistant", conclusion)

        # Calculate total execution time from agent logs
        total_time = sum(log.get("executionTimeMs", 0) for log in result.get("agentLogs", []))
        result["totalExecutionTimeMs"] = total_time

        self._save_to_db(chat_id, result)
        return result

    def _save_to_db(self, chat_id: str, result: AIActionResult):
        confidence = result['structuredResponse'].get('confidence_score', 0)
        print(f"[DB] Saved response for {chat_id}. Confidence: {confidence}")

    def clear_chat_history(self, chat_id: str):
        self.memory.clear_history(chat_id)
