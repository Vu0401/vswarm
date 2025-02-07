from .entity_memory_item import EntityMemoryItem
from ..memory import Memory
from ..storage import RAGStorage


class EntityMemory(Memory):
    """
    EntityMemory class for managing structured information about entities
    and their relationships using SQLite storage.
    Inherits from the Memory class.
    """

    def __init__(self, embedder_config=None, storage=None):
        storage = (
            storage
            if storage
            else RAGStorage(
                type="entities",
                allow_reset=True,
                embedder_config=embedder_config,
            )
        )
        super().__init__(storage)

    def save(self, item: EntityMemoryItem) -> None:  # type: ignore # BUG?: Signature of "save" incompatible with supertype "Memory"
        """Saves an entity item into the SQLite storage."""
        data = f"{item.name}({item.type}): {item.description}"
        self.storage.save(data, item.metadata)

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
            raise Exception(f"An error occurred while resetting the entity memory: {e}")
