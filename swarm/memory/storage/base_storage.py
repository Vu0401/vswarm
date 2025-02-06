from abc import ABC, abstractmethod


class BaseStorage(ABC):
    """
    Base class for storage classes.
    """

    @abstractmethod
    def _initialize_app(self):
        raise NotImplementedError
    
    @abstractmethod
    def save(self):
        raise NotImplementedError
    
    @abstractmethod
    def search(self):
        raise NotImplementedError
    
    @abstractmethod
    def reset(self):
        raise NotImplementedError   