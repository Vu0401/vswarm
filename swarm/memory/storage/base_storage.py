from abc import ABCMeta, abstractmethod


class BaseStorage(metaclass=ABCMeta):
    """
    Base class for storage classes.
    """

    @abstractmethod
    def save(self):
        raise NotImplementedError

    @abstractmethod
    def search(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
