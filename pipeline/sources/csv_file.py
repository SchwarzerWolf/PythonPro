import csv

from pipeline.sources.base import Source


class CsvSource(Source):
    def __init__(self, input_file_path: str):
        self.input_file_path = input_file_path

    def get_rows(self) -> list[dict]:
        with open(self.input_file_path, newline="", encoding="utf-8") as input_file:
            return list(csv.DictReader(input_file))
