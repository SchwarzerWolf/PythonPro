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


class Molecule:
    """Valid molecule with its calculated descriptors and fingerprint."""

    MAX_MOLECULAR_WEIGHT = 500
    MAX_LOG_P = 5
    MAX_HBD = 5
    MAX_HBA = 10

    FINGERPRINT_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)

    def __init__(self, mol_id: str, smiles: str, molecule: Chem.Mol) -> None:
        self.mol_id = mol_id
        self.smiles = smiles

        self.fingerprint = self.FINGERPRINT_GENERATOR.GetFingerprint(molecule)
        self.molecular_weight: float = Descriptors.MolWt(molecule)
        self.log_p: float = Crippen.MolLogP(molecule)
        self.tpsa: float = rdMolDescriptors.CalcTPSA(molecule)
        self.hbd: int = rdMolDescriptors.CalcNumHBD(molecule)
        self.hba: int = rdMolDescriptors.CalcNumHBA(molecule)
        self.rotatable_bonds: int = rdMolDescriptors.CalcNumRotatableBonds(molecule)
        self.rings: int = rdMolDescriptors.CalcNumRings(molecule)
        self.qed: float = Descriptors.qed(molecule)

    @classmethod
    def from_row(cls, row: dict) -> "Molecule | None":
        """Creates a molecule from a csv row, returns None if SMILES is invalid.

        SMILES is parsed only once here and the parsed object is passed to __init__.
        """
        molecule = Chem.MolFromSmiles(row["smiles"])
        if molecule is None:
            return None
        return cls(row["mol_id"], row["smiles"], molecule)

    @property
    def ro5_pass(self) -> bool:
        """Checks Lipinski's rule of five."""
        return (
            self.molecular_weight <= self.MAX_MOLECULAR_WEIGHT
            and self.log_p <= self.MAX_LOG_P
            and self.hbd <= self.MAX_HBD
            and self.hba <= self.MAX_HBA
        )

    def to_report(self) -> dict:
        """Returns fields that are saved to the output csv file."""
        return {
            "mol_id": self.mol_id,
            "smiles": self.smiles,
            "MW": self.molecular_weight,
            "LogP": self.log_p,
            "TPSA": self.tpsa,
            "HBD": self.hbd,
            "HBA": self.hba,
            "RotBonds": self.rotatable_bonds,
            "Rings": self.rings,
            "QED": self.qed,
            "Ro5_pass": self.ro5_pass,
        }


class GraphGenerator:
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

        self._rows: list[dict] = []
        self._valid_molecules: list[Molecule] = []
        self._invalid_molecules: list[dict] = []
        self._cluster_labels: list[int] = []

    def _load(self) -> None:
        """Loads data from csv file."""
        with open(self.input_file_path, newline="", encoding="utf-8") as input_file:
            self._rows = list(csv.DictReader(input_file))

    def _calculate_properties(self) -> None:
        """Calculates molecular properties from rows and separates valid and invalid records."""
        valid_molecules = []
        invalid_molecules = []
        for row in self._rows:
            molecule = Molecule.from_row(row)
            if molecule is None:
                invalid_molecules.append(row)
            else:
                valid_molecules.append(molecule)

        self._valid_molecules = valid_molecules
        self._invalid_molecules = invalid_molecules

    def _print_statistics(self) -> None:
        """Prints some statistical info."""
        total_count = len(self._valid_molecules) + len(self._invalid_molecules)
        print(f"Valid molecules: {len(self._valid_molecules)} / {total_count}")
        ro5_pass_count = sum(molecule.ro5_pass for molecule in self._valid_molecules)
        print(f"Ro5 pass: {ro5_pass_count} / {total_count}")
        molecular_weights = [molecule.molecular_weight for molecule in self._valid_molecules]
        print(f"MW range: {min(molecular_weights):.1f} — {max(molecular_weights):.1f}")
        for molecule in self._valid_molecules[:10]:
            print(
                molecule.mol_id,
                molecule.molecular_weight,
                molecule.log_p,
                molecule.tpsa,
                molecule.qed,
                molecule.ro5_pass,
            )

    def _clusterize(self) -> None:
        """Performs Butina clustering algorithm on molecules."""
        fingerprints = [molecule.fingerprint for molecule in self._valid_molecules]
        distances = []
        for i in range(1, len(fingerprints)):
            similarity_scores = BulkTanimotoSimilarity(
                fingerprints[i],
                fingerprints[:i],
            )
            distances.extend([1 - score for score in similarity_scores])
        clusters = Butina.ClusterData(distances, len(fingerprints), 0.7, isDistData=True)
        cluster_labels = [-1] * len(fingerprints)
        for cluster_id, members in enumerate(clusters):
            for idx in members:
                cluster_labels[idx] = cluster_id
        self._cluster_labels = cluster_labels

    def _plot_graph(self) -> None:
        """Creates layout and draws a graph using Faerun library."""
        fingerprints = np.array(
            [list(molecule.fingerprint) for molecule in self._valid_molecules],
            dtype=np.uint8,
        )
        x, y, edges_from, edges_to = TMAP(metric="jaccard", n_neighbors=10).fit_transform(
            fingerprints
        )

        faerun_plot = Faerun(view="front", coords=False, clear_color="#FFFFFF")
        faerun_plot.add_scatter(
            name="molecules",
            data={
                "x": x,
                "y": y,
                "c": [
                    self._cluster_labels,
                    [molecule.log_p for molecule in self._valid_molecules],
                    [molecule.molecular_weight for molecule in self._valid_molecules],
                ],
                "labels": [
                    f"{molecule.smiles}__{molecule.mol_id}" for molecule in self._valid_molecules
                ],
            },
            shader="smoothCircle",
            point_scale=10,
            colormap=["tab20", "viridis", "plasma"],
            categorical=[True, False, False],
            series_title=["Cluster", "LogP", "MolWt"],
            has_legend=True,
            legend_title=["Cluster", "LogP", "MolWt"],
        )
        faerun_plot.add_tree(
            name="molecules_tree",
            data={"from": edges_from, "to": edges_to},
            point_helper="molecules",
            color="#222222",
        )
        faerun_plot.plot("molecules", path=self.output_folder, template="smiles")

    @staticmethod
    def _write_csv(file_path: str, values: list[dict]) -> None:
        """Saves data to a specified file."""
        with open(file_path, "w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)

    def _save(self) -> None:
        """Saves valid and invalid molecules to csv files."""
        if self._invalid_molecules:
            self._write_csv(self.invalid_output_file_path, self._invalid_molecules)
        if self._valid_molecules:
            report = [molecule.to_report() for molecule in self._valid_molecules]
            self._write_csv(self.output_file_path, report)

    def run(self) -> None:
        """Orchestrates the pipeline."""
        self._load()
        self._calculate_properties()
        if not self._valid_molecules:
            print("No valid molecules found, nothing to process.")
            self._save()
            return
        self._print_statistics()
        self._clusterize()
        self._plot_graph()
        self._save()


if __name__ == "__main__":
    input_file_path = os.path.join("data", "molecules.csv")
    output_file_folder = os.path.join("output")
    graph_generator = GraphGenerator(
        input_file_path,
        output_file_folder,
    )
    graph_generator.run()
