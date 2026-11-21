"""Пайплайн для процессинга молекул.

Читает молекулы из файла, считает дескрипторы и фингерпринты,
делает кластеризацию и визуализирует результаты.
"""

import csv
import os

import numpy as np
from faerun import Faerun
from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    rdFingerprintGenerator,
    rdMolDescriptors,
)
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from tmap import TMAP


class GraphGenerator:
    MAX_MOLECULAR_WEIGHT = 500
    MAX_LOG_P = 5
    MAX_HDB = 5
    MAX_HBA = 10

    REPORT_FIELDS = {
        "mol_id",
        "smiles",
        "MW",
        "LogP",
        "TPSA",
        "HBD",
        "HBA",
        "RotBonds",
        "Rings",
        "QED",
        "Ro5_pass",
    }

    VALID_FILE_OUTPUT_NAME = "molecules_processed.csv"
    INVALID_FILE_OUTPUT_NAME = "molecules_invalid.csv"

    def __init__(self, input_file_path: str, output_folder: str) -> None:
        self.input_file_path: str = input_file_path
        self.output_folder: str = output_folder

        self.output_file_path = os.path.join(
            output_folder,
            self.VALID_FILE_OUTPUT_NAME,
        )
        self.invalid_output_file_path = os.path.join(
            output_folder,
            self.INVALID_FILE_OUTPUT_NAME,
        )
        os.makedirs(self.output_folder, exist_ok=True)

        self._rows = None
        self._valid_molecules = None
        self._invalid_molecules = None

    def _load(self) -> None:
        """Loads data from csv file."""
        with open(self.input_file_path, newline="", encoding="utf-8") as input_file:
            self._rows = list(csv.DictReader(input_file))

    def _calculate_properties(self) -> None:
        """Calculates molecular properties from rows and separates valid and invalid records."""
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)

        valid_molecules = []
        invalid_molecules = []
        for row in self._rows:
            molecule = Chem.MolFromSmiles(row["smiles"])
            if molecule is None:
                invalid_molecules.append(row)
            else:
                molecular_properties = {
                    **row,
                    "molecule": molecule,
                    "fingerprint": generator.GetFingerprint(molecule),
                    "MW": Descriptors.MolWt(molecule),
                    "LogP": Crippen.MolLogP(molecule),
                    "TPSA": rdMolDescriptors.CalcTPSA(molecule),
                    "HBD": rdMolDescriptors.CalcNumHBD(molecule),
                    "HBA": rdMolDescriptors.CalcNumHBA(molecule),
                    "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(molecule),
                    "Rings": rdMolDescriptors.CalcNumRings(molecule),
                    "QED": Descriptors.qed(molecule),
                }
                molecular_properties["Ro5_pass"] = (
                    (molecular_properties["MW"] <= self.MAX_MOLECULAR_WEIGHT)
                    and (molecular_properties["LogP"] <= self.MAX_LOG_P)
                    and (molecular_properties["HBD"] <= self.MAX_HDB)
                    and (molecular_properties["HBA"] <= self.MAX_HBA)
                )
                valid_molecules.append(molecular_properties)

        self._valid_molecules = valid_molecules
        self._invalid_molecules = invalid_molecules


    def _print_statistics(self) -> None:
        """Prints some statistical info."""
        total_count = len(self._valid_molecules) + len(self._invalid_molecules)
        print(f"Valid molecules: {len(self._valid_molecules)} / {total_count}")
        print(f"Ro5 pass: {sum(row['Ro5_pass'] for row in self._valid_molecules)} / {total_count}")
        molecular_weights = [row["MW"] for row in self._valid_molecules]
        print(f"MW range: {min(molecular_weights):.1f} — {max(molecular_weights):.1f}")
        for row in self._valid_molecules[:10]:
            print(row["mol_id"], row["MW"], row["LogP"], row["TPSA"], row["QED"], row["Ro5_pass"])


    def _clusterize(self) -> None:
        """Performs Butina clustering algorithm on molecules."""
        fingerprints = [molecule["fingerprint"] for molecule in self._valid_molecules]
        distances = []
        for i in range(1, len(fingerprints)):
            similarity_scores = BulkTanimotoSimilarity(
                fingerprints[i],
                fingerprints[:i],
            )
            distances.extend([1 - score for score in similarity_scores])
        clusters = Butina.ClusterData(distances, len(fingerprints), 0.7, isDistData=True)
        for row in self._valid_molecules:
            row["cluster"] = -1
        for cluster_id, members in enumerate(clusters):
            for idx in members:
                self._valid_molecules[idx]["cluster"] = cluster_id


    def _plot_graph(self) -> None:
        """Creates layout and draws a graph using Faerun library."""
        fingerprints = np.array(
            [list(molecule["fingerprint"]) for molecule in self._valid_molecules],
            dtype=np.uint8,
        )
        x, y, s, t = TMAP(metric="jaccard", n_neighbors=10).fit_transform(fingerprints)
        for row, xi, yi in zip(self._valid_molecules, x, y, strict=True):
            row["x"] = xi
            row["y"] = yi

        f = Faerun(view="front", coords=False, clear_color="#FFFFFF")
        f.add_scatter(
            name="molecules",
            data={
                "x": x,
                "y": y,
                "c": [
                    [row["cluster"] for row in self._valid_molecules],
                    [row["LogP"] for row in self._valid_molecules],
                    [row["MW"] for row in self._valid_molecules],
                ],
                "labels": [f"{row['smiles']}__{row['mol_id']}" for row in self._valid_molecules],
            },
            shader="smoothCircle",
            point_scale=10,
            colormap=["tab20", "viridis", "plasma"],
            categorical=[True, False, False],
            series_title=["Cluster", "LogP", "MolWt"],
            has_legend=True,
            legend_title=["Cluster", "LogP", "MolWt"],
        )
        f.add_tree(
            name="molecules_tree",
            data={"from": s, "to": t},
            point_helper="molecules",
            color="#222222",
        )
        f.plot("molecules", path=self.output_folder, template="smiles")


    def _filter_output(self, molecules: list[dict]) -> list[dict]:
        """Filters record info only to required fields."""
        return [
            {key: value for key, value in molecule.items() if key in self.REPORT_FIELDS}
            for molecule in molecules
        ]

    @staticmethod
    def _write_csv(file_path: str, values: list[dict]) -> None:
        """Saves data to a specified file."""
        with open(file_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    def _save(self):
        if self._invalid_molecules:
            self._write_csv(self.invalid_output_file_path, self._invalid_molecules)
        filtered_values = self._filter_output(self._valid_molecules)
        self._write_csv(self.output_file_path, filtered_values)

    def run(self) -> None:
        """Orchestrates the pipeline."""
        self._load()
        self._calculate_properties()
        # TODO check if there are any molecules
        self._print_statistics()
        self._clusterize()
        self._plot_graph()
        self._save()


if __name__ == "__main__":
    input_file_path = os.path.join("data", "molecules.csv")
    output_file_folder = os.path.join("output")
    generator = GraphGenerator(input_file_path, output_file_folder)
    generator.run()
    # способ обработки данных к данным
