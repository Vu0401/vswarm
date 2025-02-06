from typing import Any, Dict, Optional

from .short_term_memory_item import ShortTermMemoryItem
from ..storage import RAGStorage
from ..memory import Memory


class ShortTermMemory(Memory):
    """
    ShortTermMemory class for managing transient data related to immediate tasks
    and interactions.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    MemoryItem instances.
    """

    def __init__(self, embedder_config=None, storage=None, allow_reset=None):
        storage = (
            storage if storage else
            RAGStorage(
                type="short_term",
                embedder_config=embedder_config,
                allow_reset=allow_reset,
            )
        )
        super().__init__(storage)

    def save(
        self,
        item: ShortTermMemoryItem
    ) -> None:
        # update data/metadata here (if needed)
        self.storage.save(
            value=item.data,
            metadata=item.metadata,
        )

    def search(
        self,
        query: str,
        limit: int = 3,
        score_threshold: float = 0.35,
    ):
        return self.storage.search(
            query=query, limit=limit, score_threshold=score_threshold
        ) 

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(
                f"An error occurred while resetting the short-term memory: {e}"
            )
