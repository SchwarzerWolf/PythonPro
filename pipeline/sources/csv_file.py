"""Molecule source that reads a csv file."""

import csv

from pipeline.sources.base import Source


class CsvSource(Source):
    """Reads molecules from a csv file with mol_id and smiles columns."""

    def __init__(self, input_file_path: str) -> None:
        """Creates the source.

        Args:
            input_file_path: csv file with mol_id and smiles columns.
        """
        self.input_file_path = input_file_path

    def get_rows(self) -> list[dict]:
        """Reads rows from the csv file."""
        with open(self.input_file_path, newline="", encoding="utf-8") as input_file:
            return list(csv.DictReader(input_file))
