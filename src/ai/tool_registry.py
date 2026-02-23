from typing import List, Dict, Any, Callable
from .retriever_service import RetrieverService
from .conflict_detector import ConflictDetector

class ToolRegistry:
    def __init__(self, retriever: RetrieverService):
        self.retriever = retriever
        self.tools: Dict[str, Dict[str, Any]] = {}
        self._register_tools()

    def _register_tools(self):
        self.tools["search_legal_database"] = {
            "name": "search_legal_database",
            "description": "Search for relevant IPC/CrPC sections and cases.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"]
            },
            "execute": lambda args: self.retriever.retrieve(args["query"])
        }

        self.tools["detect_conflicts"] = {
            "name": "detect_conflicts",
            "description": "Analyze retrieved documents for contradictions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "documents": {"type": "array", "items": {"type": "object"}}
                }
            },
            "execute": lambda args: ConflictDetector.detect(args["documents"])
        }

    def get_definitions(self) -> List[Dict[str, Any]]:
        return [
            {k: v for k, v in tool.items() if k != "execute"}
            for tool in self.tools.values()
        ]

    def execute_tool(self, name: str, args: Any) -> Any:
        if name not in self.tools:
            raise ValueError(f"Tool {name} not found")
        return self.tools[name]["execute"](args)
