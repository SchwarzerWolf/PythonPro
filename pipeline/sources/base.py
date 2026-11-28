"""Common interface of molecule sources."""

from abc import ABC, abstractmethod


class Source(ABC):
    """Source of input rows with mol_id and smiles columns."""

    @abstractmethod
    def get_rows(self) -> list[dict]:
        """Returns rows with mol_id and smiles."""
        pass
