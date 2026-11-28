"""Пайплайн для процессинга молекул.

Берёт молекулы из выбранного источника, считает дескрипторы и фингерпринты,
делает кластеризацию и визуализирует результаты.
"""

import argparse

from pipeline.graph_generator import GraphGenerator
from pipeline.sources import CsvSource, ReinventSource, Source


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(description="Molecule processing pipeline")
    parser.add_argument("--output", default="output", help="folder for results")
    subparsers = parser.add_subparsers(dest="source", required=True)

    csv_parser = subparsers.add_parser("csv", help="read molecules from a csv file")
    csv_parser.add_argument("input_file_path", help="csv file with mol_id and smiles columns")

    subparsers.add_parser("reinvent", help="sample molecules with REINVENT")

    return parser.parse_args()


def create_source(args: argparse.Namespace) -> Source:
    """Creates a molecule source chosen in command line arguments."""
    if args.source == "csv":
        return CsvSource(args.input_file_path)
    if args.source == "reinvent":
        return ReinventSource()
    raise NotImplementedError("No builder was defined for the chosen source.")


if __name__ == "__main__":
    args = parse_args()
    source = create_source(args)
    graph_generator = GraphGenerator(source, args.output)
    graph_generator.run()
