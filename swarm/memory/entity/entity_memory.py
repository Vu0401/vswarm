from .entity_memory_item import EntityMemoryItem
from ..memory import Memory
import datetime
from swarm.memory.storage import RAGStorage


class EntityMemory(Memory):
    def __init__(self, agent_name: str, chroma_client=None):
        collection_name = f"{agent_name}_entity_memory"
        storage = RAGStorage(collection_name=collection_name,
                             chroma_client=chroma_client)
        super().__init__(agent_name=agent_name, storage=storage)

    def save(self, entity: EntityMemoryItem) -> None:
        """Save an entity to memory"""
        document = f"{entity.type}: {entity.name}\n{entity.description}"
        metadata = {
            "agent": self.agent_name,
            "name": entity.name,
            "type": entity.type,
            "timestamp": datetime.datetime.now().isoformat(),
            "relationships": entity.metadata["relationships"],
        }

        entity_id = str(entity.name)

        try:
            self.storage.save(
                key=entity_id,
                value=document,
                metadata=metadata,
            )
        except Exception as e:
            raise Exception(f"Failed to save entity {entity.name}: {str(e)}")

    def search(
        self,
        query: str,
        limit: int = 3,
    ):
        return self.storage.search(
            text=query, 
            limit=limit,
            metadata={"agent": self.agent_name}
        )

    def reset(self) -> None:
        try:
            self.storage.reset()
        except Exception as e:
            raise Exception(f"An error occurred while resetting the entity memory: {e}")
