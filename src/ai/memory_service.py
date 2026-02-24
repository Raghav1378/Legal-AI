from typing import List, Dict, Any

class MemoryService:
    def __init__(self):
        self._storage: Dict[str, List[Dict[str, str]]] = {}
        self._results: Dict[str, List[Dict[str, Any]]] = {}

    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        return self._storage.get(chat_id, [])

    def add_message(self, chat_id: str, role: str, content: str):
        if chat_id not in self._storage:
            self._storage[chat_id] = []
        self._storage[chat_id].append({"role": role, "content": content})

    def add_result(self, chat_id: str, result: Dict[str, Any]):
        if chat_id not in self._results:
            self._results[chat_id] = []
        self._results[chat_id].append(result)

    def get_results(self, chat_id: str) -> List[Dict[str, Any]]:
        return self._results.get(chat_id, [])

    def clear_history(self, chat_id: str):
        if chat_id in self._storage:
            self._storage[chat_id] = []
        if chat_id in self._results:
            self._results[chat_id] = []
