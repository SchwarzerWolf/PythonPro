"""Pipeline that processes molecules, clusters them and draws a graph."""

import csv
import os

import numpy as np
from faerun import Faerun
from rdkit.DataStructs.cDataStructs import BulkTanimotoSimilarity
from rdkit.ML.Cluster import Butina
from tmap import TMAP

from pipeline.molecule import Molecule
from pipeline.sources import Source


class GraphGenerator:
    """Processes molecules from a source: descriptors, clustering, graph and csv reports."""

    VALID_FILE_OUTPUT_NAME = "molecules_processed.csv"
    INVALID_FILE_OUTPUT_NAME = "molecules_invalid.csv"
    CLUSTER_DISTANCE_THRESHOLD = 0.7  # Butina: молекулы ближе этого расстояния – в одном кластере
    N_NEIGHBORS = 10  # TMAP: сколько соседей учитывать для каждой молекулы

    def __init__(self, source: Source, output_folder: str) -> None:
        """Creates the pipeline.

        Args:
            source: where to get molecules from.
            output_folder: folder for csv reports and the html graph.
        """
        self.source = source
        self.output_folder = output_folder

        self.output_file_path = os.path.join(output_folder, self.VALID_FILE_OUTPUT_NAME)
        self.invalid_output_file_path = os.path.join(output_folder, self.INVALID_FILE_OUTPUT_NAME)
        os.makedirs(self.output_folder, exist_ok=True)

        self._rows: list[dict] = []
        self._valid_molecules: list[Molecule] = []
        self._invalid_molecules: list[dict] = []
        self._cluster_labels: list[int] = []

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
        clusters = Butina.ClusterData(
            distances,
            len(fingerprints),
            self.CLUSTER_DISTANCE_THRESHOLD,
            isDistData=True,
        )
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
        # соседей не может быть больше, чем остальных молекул
        n_neighbors = min(self.N_NEIGHBORS, len(fingerprints) - 1)
        x, y, edges_from, edges_to = TMAP(
            metric="jaccard",
            n_neighbors=n_neighbors,
        ).fit_transform(fingerprints)

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
        self._rows = self.source.get_rows()
        self._calculate_properties()
        if not self._valid_molecules:
            print("No valid molecules found, nothing to process.")
            self._save()
            return
        self._print_statistics()
        self._clusterize()
        self._plot_graph()
        self._save()
