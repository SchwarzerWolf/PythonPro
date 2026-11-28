from rdkit import Chem
from rdkit.Chem import (
    Crippen,
    Descriptors,
    rdFingerprintGenerator,
    rdMolDescriptors,
)


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



