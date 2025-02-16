from typing import Any, Dict, List

from .long_term_memory_item import LongTermMemoryItem
from ..memory import Memory
from ..storage import SQLiteStorage


class LongTermMemory(Memory):
    """
    LongTermMemory class for managing cross runs data related to overall crew's
    execution and performance.
    Inherits from the Memory class and utilizes an instance of a class that
    adheres to the Storage for data storage, specifically working with
    LongTermMemoryItem instances.
    """

    def __init__(self, agent_name: str, storage=None, sql_connection_string=None):
        if not storage:
            storage = (
                SQLiteStorage(db_path=sql_connection_string)
                if sql_connection_string
                else SQLiteStorage()
            )
        super().__init__(agent_name=agent_name, storage=storage)

    def save(self, item: LongTermMemoryItem) -> None:
        metadata = item.metadata
        metadata.update({"agent": item.agent, "expected_output": item.expected_output})
        self.storage.save(
            task_description=item.task,
            score=metadata["quality"],
            metadata=metadata,
            datetime=item.datetime,
        )

    def search(self, task: str, latest_n: int = 3) -> List[Dict[str, Any]]:
        return self.storage.search(task, latest_n)

    def reset(self) -> None:
        self.storage.reset()
