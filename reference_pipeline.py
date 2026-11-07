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


with open(INPUT_FILE_PATH, newline="", encoding="utf-8") as input_file:
    rows = list(csv.DictReader(input_file))


mw_list = []
logp_list = []
tpsa_list = []
hbd_list = []
hba_list = []
rotbonds_list = []
rings_list = []
qed_list = []
valid_list = []
molecules = [Chem.MolFromSmiles(row["smiles"]) for row in rows]
for smiles in [row["smiles"] for row in rows]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        mw_list.append(Descriptors.MolWt(molecule))
        logp_list.append(Descriptors.MolLogP(molecule))
        tpsa_list.append(Descriptors.TPSA(molecule))
        hbd_list.append(Descriptors.NumHDonors(molecule))
        hba_list.append(Descriptors.NumHAcceptors(molecule))
        rotbonds_list.append(Descriptors.NumRotatableBonds(molecule))
        rings_list.append(Descriptors.RingCount(molecule))
        qed_list.append(Descriptors.qed(molecule))
        valid_list.append(True)
    else:
        mw_list.append(None)
        logp_list.append(None)
        tpsa_list.append(None)
        hbd_list.append(None)
        hba_list.append(None)
        rotbonds_list.append(None)
        rings_list.append(None)
        qed_list.append(None)
        valid_list.append(False)

for i, row in enumerate(rows):
    row["valid"] = valid_list[i]
    row["MW"] = mw_list[i]
    row["LogP"] = logp_list[i]
    row["TPSA"] = tpsa_list[i]
    row["HBD"] = hbd_list[i]
    row["HBA"] = hba_list[i]
    row["RotBonds"] = rotbonds_list[i]
    row["Rings"] = rings_list[i]
    row["QED"] = qed_list[i]
    row["Ro5_pass"] = (
        (row["MW"] <= MAX_MOLECULAR_WEIGHT)
        and (row["LogP"] <= MAX_LOG_P)
        and (row["HBD"] <= MAX_HDB)
        and (row["HBA"] <= MAX_HBA)
    )

# print statistics
print(f"Valid molecules: {sum(valid_list)} / {len(rows)}")
print(f"Ro5 pass: {sum(row['Ro5_pass'] for row in rows)} / {len(rows)}")
print(f"MW range: {min(mw_list):.1f} — {max(mw_list):.1f}")
for row in rows[:10]:
    print(row["mol_id"], row["MW"], row["LogP"], row["TPSA"], row["QED"], row["Ro5_pass"])

generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
fingerprints = [
    generator.GetFingerprint(molecule)
    for molecule in molecules
]
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
for row in rows:
    row["cluster"] = -1
for cluster_id, members in enumerate(clusters):
    for idx in members:
        rows[idx]["cluster"] = cluster_id

X = np.array([list(fingerprint) for fingerprint in fingerprints], dtype=np.uint8)
x, y, s, t = TMAP(metric="jaccard", n_neighbors=10).fit_transform(X)
for row, xi, yi in zip(rows, x, y, strict=True):
    row["x"] = xi
    row["y"] = yi

os.makedirs(OUTPUT_FILE_PATH, exist_ok=True)
f = Faerun(view="front", coords=False, clear_color="#FFFFFF")
f.add_scatter(
    name="molecules",
    data={
        "x": x,
        "y": y,
        "c": [
            [row["cluster"] for row in rows],
            [row["LogP"] for row in rows],
            [row["MW"] for row in rows],
        ],
        "labels": [
            f"{row['smiles']}__{row['mol_id']}"
            for row in rows
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
f.add_tree(
    name="molecules_tree",
    data={"from": s, "to": t},
    point_helper="molecules",
    color="#222222",
)
f.plot("molecules", path=OUTPUT_FILE_FOLDER, template="smiles")

with open(OUTPUT_FILE_PATH, "w", newline="", encoding="utf-8") as output_file:
    writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
