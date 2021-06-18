from abc import ABC, abstractmethod


class Replay(ABC):

    def __init__(self, size):
        pass

    @abstractmethod
    def add_transition(self, x):
        pass

    @abstractmethod
    def get_batch(self, b):
        pass
