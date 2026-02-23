from typing import List, Dict

class MemoryService:
    def __init__(self):
        # In-memory storage for MVP, easy to swap for Postgres/Redis
        self._storage: Dict[str, List[Dict[str, str]]] = {}

    def get_history(self, chat_id: str) -> List[Dict[str, str]]:
        return self._storage.get(chat_id, [])

    def add_message(self, chat_id: str, role: str, content: str):
        if chat_id not in self._storage:
            self._storage[chat_id] = []
        self._storage[chat_id].append({"role": role, "content": content})

    def clear_history(self, chat_id: str):
        if chat_id in self._storage:
            self._storage[chat_id] = []
