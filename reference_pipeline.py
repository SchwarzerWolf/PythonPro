"""Пайплайн для процессинга молекул.

Читает молекулы из файла, считает дескрипторы и фингерпринты,
делает кластеризацию и визуализирует результаты.
"""

import csv
import os

import numpy as np
from faerun import Faerun
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
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


def load(input_file_path):
    with open(input_file_path, newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def calculate_properties(rows):
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
                "LogP": Descriptors.MolLogP(molecule),
                "TPSA": Descriptors.TPSA(molecule),
                "HBD": Descriptors.NumHDonors(molecule),
                "HBA": Descriptors.NumHAcceptors(molecule),
                "RotBonds": Descriptors.NumRotatableBonds(molecule),
                "Rings": Descriptors.RingCount(molecule),
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


def print_statistics(valid_molecules, invalid_molecules):
    total_count = len(valid_molecules) + len(invalid_molecules)
    print(f"Valid molecules: {len(valid_molecules)} / {total_count}")
    print(f"Ro5 pass: {sum(row['Ro5_pass'] for row in valid_molecules)} / {total_count}")
    molecular_weights = [row["MW"] for row in valid_molecules]
    print(f"MW range: {min(molecular_weights):.1f} — {max(molecular_weights):.1f}")
    for row in valid_molecules[:10]:
        print(row["mol_id"], row["MW"], row["LogP"], row["TPSA"], row["QED"], row["Ro5_pass"])


def clusterize(valid_molecules):
    fingerprints = [molecule["fingerprint"] for molecule in valid_molecules]
    distances = []
    for i in range(1, len(fingerprints)):
        similarity_scores = DataStructs.BulkTanimotoSimilarity(
            fingerprints[i],
            fingerprints[:i],
        )
        distances.extend([1 - score for score in similarity_scores])
    clusters = Butina.ClusterData(
        distances,
        len(fingerprints),
        0.7,
        isDistData=True
    )
    for row in valid_molecules:
        row["cluster"] = -1
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            valid_molecules[idx]["cluster"] = cluster_id


def plot_graph(valid_molecules):
    fingerprints = [molecule["fingerprint"] for molecule in valid_molecules]
    X = np.array([list(fingerprint) for fingerprint in fingerprints], dtype=np.uint8)
    x, y, s, t = TMAP(metric="jaccard", n_neighbors=10).fit_transform(X)
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


def filter_output(molecules):
    return [
        {
            key: value for key, value in molecule.items()
            if key in REPORT_FIELDS
        } for molecule in molecules
    ]


def save(file_path, values):
    with open(file_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def run():
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
