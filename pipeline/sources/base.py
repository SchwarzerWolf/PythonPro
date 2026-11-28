from abc import abstractmethod, ABC


class Source(ABC):
    @abstractmethod
    def get_rows(self) -> list[dict]:
        pass