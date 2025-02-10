from abc import ABC, abstractmethod
from .storage.base_storage import BaseStorage
from typing import Any, Dict, List, Optional

class Memory(ABC):
    """
    Base class for memory, now supporting agent tags and generic metadata.
    """

    def __init__(self, storage):
        if not isinstance(storage, BaseStorage):
            raise ValueError("storage must be an instance of BaseStorage")
        self.storage = storage

    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def search(self):
        raise NotImplementedError
