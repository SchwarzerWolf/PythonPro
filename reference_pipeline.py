from faerun import Faerun
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from tmap import TMAP
import csv
import os
import numpy as np


mw_list = []
logp_list = []
tpsa_list = []
hbd_list = []
hba_list = []
rotbonds_list = []
rings_list = []
qed_list = []
valid_list = []
os.makedirs("output", exist_ok=True)
with open("data/molecules.csv", newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
# process all molecules fast, don't touch
mols = [Chem.MolFromSmiles(r["smiles"]) for r in rows]
for smi in [r["smiles"] for r in rows]:
    mol = Chem.MolFromSmiles(smi)
    print("ok")
    if mol is not None:
        mw_list.append(Descriptors.MolWt(mol))
        logp_list.append(Descriptors.MolLogP(mol))
        tpsa_list.append(Descriptors.TPSA(mol))
        hbd_list.append(Descriptors.NumHDonors(mol))
        hba_list.append(Descriptors.NumHAcceptors(mol))
        rotbonds_list.append(Descriptors.NumRotatableBonds(mol))
        rings_list.append(Descriptors.RingCount(mol))
        qed_list.append(Descriptors.qed(mol))
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

for i, r in enumerate(rows):
    r["valid"] = valid_list[i]
    r["MW"] = mw_list[i]
    r['LogP'] = logp_list[i]
    r['TPSA'] = tpsa_list[i]
    r["HBD"] = hbd_list[i]
    r["HBA"] = hba_list[i]
    r["RotBonds"] = rotbonds_list[i]
    r["Rings"] = rings_list[i]
    r["QED"] = qed_list[i]
    r["Ro5_pass"] = (r["MW"] <= 500) and (r["LogP"] <= 5) and (r["HBD"] <= 5) and (r["HBA"] <= 10)
# print statistics
print(f"Valid molecules: {sum(valid_list)} / {len(rows)}")
print(f"Ro5 pass: {sum(r['Ro5_pass'] for r in rows)} / {len(rows)}")
print(f"MW range: {min(mw_list):.1f} — {max(mw_list):.1f}")
for r in rows[:10]:
    print(r["mol_id"], r["MW"], r["LogP"], r["TPSA"], r["QED"], r["Ro5_pass"])
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
fps = [gen.GetFingerprint(m) for m in mols]
distances = []
for i in range(1, len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    distances.extend([1 - s for s in sims])
clusters = Butina.ClusterData(distances, len(fps), 0.7, isDistData=True)
for r in rows:
    r['cluster'] = -1
for cluster_id, members in enumerate(clusters):
    for idx in members:
        rows[idx]['cluster'] = cluster_id
X = np.array([list(fp) for fp in fps], dtype=np.uint8)
x, y, s, t = TMAP(metric="jaccard", n_neighbors=10).fit_transform(X)
for r, xi, yi in zip(rows, x, y):
    r["x"] = xi
    r["y"] = yi
with open("output/molecules_new.csv", "w", newline="", encoding="utf-8") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
labels = [f"{r['smiles']}__{r['mol_id']}" for r in rows]
# use Faerun
f = Faerun(view="front", coords=False, clear_color="#FFFFFF")
f.add_scatter(
    "molecules", { "x": x, "y": y, "c": [[r["cluster"] for r in rows], [r["LogP"] for r in rows], [r["MW"] for r in rows]], "labels": labels},
    shader="smoothCircle", point_scale=10, colormap=["tab20", "viridis", "plasma"], categorical=[True, False, False],
    series_title=["Cluster", "LogP", "MolWt"], has_legend=True, legend_title=["Cluster", "LogP", "MolWt"]
)
f.add_tree(
    "molecules_tree", {"from": s, "to": t}, point_helper="molecules", color="#222222",
)
f.plot("molecules", path="output", template="smiles")
# check the values in the browser








