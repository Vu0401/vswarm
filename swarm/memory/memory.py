from .storage.base_storage import BaseStorage
from typing import Any
from pydantic import BaseModel, InstanceOf


class Memory(BaseModel):
    agent_name: str
    storage: InstanceOf[BaseStorage]

    def save(self, item: Any) -> None:
        raise NotImplementedError

    def retrieve(self, query: str) -> Any:
        raise NotImplementedError

    def update(self, key: str, value: Any) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> None:
        raise NotImplementedError
