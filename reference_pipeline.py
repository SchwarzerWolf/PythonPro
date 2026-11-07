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


with open(INPUT_FILE_PATH, newline="", encoding="utf-8") as input_file:
    rows = list(csv.DictReader(input_file))

valid_molecules = []
invalid_molecules = []
generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
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

if invalid_molecules:
    with open(INVALID_OUTPUT_FILE_PATH, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(invalid_molecules[0]))
        writer.writeheader()
        writer.writerows(invalid_molecules)

# print statistics
print(f"Valid molecules: {len(valid_molecules)} / {len(rows)}")
print(f"Ro5 pass: {sum(row['Ro5_pass'] for row in valid_molecules)} / {len(rows)}")
molecular_weights = [row["MW"] for row in valid_molecules]
print(f"MW range: {min(molecular_weights):.1f} — {max(molecular_weights):.1f}")
for row in valid_molecules[:10]:
    print(row["mol_id"], row["MW"], row["LogP"], row["TPSA"], row["QED"], row["Ro5_pass"])

fingerprints = [molecule["fingerprint"] for molecule in valid_molecules]
distances = []
for i in range(1, len(fingerprints)):
    similarity_scores = DataStructs.BulkTanimotoSimilarity(
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

X = np.array([list(fingerprint) for fingerprint in fingerprints], dtype=np.uint8)
x, y, s, t = TMAP(metric="jaccard", n_neighbors=10).fit_transform(X)
for row, xi, yi in zip(valid_molecules, x, y, strict=True):
    row["x"] = xi
    row["y"] = yi

os.makedirs(OUTPUT_FILE_FOLDER, exist_ok=True)
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

for molecule in valid_molecules:
    del molecule["molecule"]
    del molecule["fingerprint"]

with open(OUTPUT_FILE_PATH, "w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=list(valid_molecules[0]))
    writer.writeheader()
    writer.writerows(valid_molecules)
