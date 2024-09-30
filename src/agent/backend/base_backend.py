from abc import abstractmethod


class Backend:
    def __init__(
            self,
            name: str
    ):
        self.name = name
        self.messages = []

    @abstractmethod
    def generate(self, message: str):
        raise NotImplementedError("class must implement error")


