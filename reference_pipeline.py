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

MAX_MOLECULAR_WEIGHT = 500
MAX_LOG_P = 5
MAX_HDB = 5
MAX_HBA = 10

INPUT_FILE_PATH = os.path.join("data", "molecules.csv")
OUTPUT_FILE_FOLDER = os.path.join("output")
OUTPUT_FILE_PATH = os.path.join(
    OUTPUT_FILE_FOLDER,
    "molecules_processed.csv",
)
INVALID_OUTPUT_FILE_PATH = os.path.join(
    OUTPUT_FILE_FOLDER,
    "molecules_invalid.csv",
)

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


def load(input_file_path: str) -> list[dict]:
    """Loads data from csv file."""
    with open(input_file_path, newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def calculate_properties(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Calculates molecular properties from rows and separates valid and invalid records."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)

    valid_molecules = []
    invalid_molecules = []
    for row in rows:
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
                (molecular_properties["MW"] <= MAX_MOLECULAR_WEIGHT)
                and (molecular_properties["LogP"] <= MAX_LOG_P)
                and (molecular_properties["HBD"] <= MAX_HDB)
                and (molecular_properties["HBA"] <= MAX_HBA)
            )
            valid_molecules.append(molecular_properties)

    return valid_molecules, invalid_molecules


def print_statistics(valid_molecules: list[dict], invalid_molecules: list[dict]) -> None:
    """Prints some statistical info."""
    total_count = len(valid_molecules) + len(invalid_molecules)
    print(f"Valid molecules: {len(valid_molecules)} / {total_count}")
    print(f"Ro5 pass: {sum(row['Ro5_pass'] for row in valid_molecules)} / {total_count}")
    molecular_weights = [row["MW"] for row in valid_molecules]
    print(f"MW range: {min(molecular_weights):.1f} — {max(molecular_weights):.1f}")
    for row in valid_molecules[:10]:
        print(row["mol_id"], row["MW"], row["LogP"], row["TPSA"], row["QED"], row["Ro5_pass"])


def clusterize(valid_molecules: list[dict]) -> None:
    """Performs Butina clustering algorithm on molecules."""
    fingerprints = [molecule["fingerprint"] for molecule in valid_molecules]
    distances = []
    for i in range(1, len(fingerprints)):
        similarity_scores = BulkTanimotoSimilarity(
            fingerprints[i],
            fingerprints[:i],
        )
        distances.extend([1 - score for score in similarity_scores])
    clusters = Butina.ClusterData(distances, len(fingerprints), 0.7, isDistData=True)
    for row in valid_molecules:
        row["cluster"] = -1
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            valid_molecules[idx]["cluster"] = cluster_id


def plot_graph(valid_molecules: list[dict]) -> None:
    """Creates layout and draws a graph using Faerun library."""
    fingerprints = np.array(
        [list(molecule["fingerprint"]) for molecule in valid_molecules],
        dtype=np.uint8,
    )
    x, y, s, t = TMAP(metric="jaccard", n_neighbors=10).fit_transform(fingerprints)
    for row, xi, yi in zip(valid_molecules, x, y, strict=True):
        row["x"] = xi
        row["y"] = yi

    f = Faerun(view="front", coords=False, clear_color="#FFFFFF")
    f.add_scatter(
        name="molecules",
        data={
            "x": x,
            "y": y,
            "c": [
                [row["cluster"] for row in valid_molecules],
                [row["LogP"] for row in valid_molecules],
                [row["MW"] for row in valid_molecules],
            ],
            "labels": [f"{row['smiles']}__{row['mol_id']}" for row in valid_molecules],
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
    f.plot("molecules", path=OUTPUT_FILE_FOLDER, template="smiles")


def filter_output(molecules: list[dict]) -> list[dict]:
    """Filters record info only to required fields."""
    return [
        {key: value for key, value in molecule.items() if key in REPORT_FIELDS}
        for molecule in molecules
    ]


def save(file_path: str, values: list[dict]) -> None:
    """Saves data to a specified file."""
    with open(file_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def run() -> None:
    """Orchestrates the pipeline."""
    os.makedirs(OUTPUT_FILE_FOLDER, exist_ok=True)
    molecules = load(INPUT_FILE_PATH)
    valid_molecules, invalid_molecules = calculate_properties(molecules)

    if invalid_molecules:
        save(INVALID_OUTPUT_FILE_PATH, invalid_molecules)
    print_statistics(valid_molecules, invalid_molecules)

    clusterize(valid_molecules)
    plot_graph(valid_molecules)

    filtered_values = filter_output(valid_molecules)
    save(OUTPUT_FILE_PATH, filtered_values)


if __name__ == "__main__":
    run()
