from typing import List, Dict
from .short_term_memory_item import ShortTermMemoryItem
from ..storage import RAGStorage
from ..memory import Memory

class ShortTermMemory(Memory):
    def __init__(self, agent_name: str, chroma_client=None):
        collection_name = f"{agent_name}_short_term"
        storage = RAGStorage(collection_name=collection_name,
                             chroma_client=chroma_client)
        super().__init__(agent_name=agent_name, storage=storage)

    def save(self, item: ShortTermMemoryItem) -> None:
        # TODO: generate key mechanism should be improved
        key = f"{self.agent_name} {item.data}"
        item.metadata['agent'] = self.agent_name
        self.storage.save(key, item.data, item.metadata)

    def search(self, query: str, limit=10) -> List[Dict]:
        # Add agent_name to metadata filter
        return self.storage.search(
            text=query,
            limit=limit,
            metadata={"agent": self.agent_name},
        )

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the short-term memory: {e}"
            )
