"""Molecule source that samples molecules with REINVENT."""

import csv
import json
import os
import subprocess
import tempfile

from pipeline.sources.base import Source


class ReinventSource(Source):
    """Samples molecules with REINVENT.

    Paths to REINVENT are taken from the environment variables
    REINVENT_EXECUTABLE and REINVENT_MODEL_FILE (see .env.example).
    """

    CONFIG_FILE_NAME = "sampling.json"
    OUTPUT_FILE_NAME = "sampling.csv"
    NUM_SMILES = 30
    DEVICE = "cpu"

    def __init__(self) -> None:
        """Reads REINVENT paths from the environment, fails early if they are not set."""
        self.executable = os.environ["REINVENT_EXECUTABLE"]
        self.model_file = os.environ["REINVENT_MODEL_FILE"]

    def get_rows(self) -> list[dict]:
        """Runs REINVENT sampling in a temporary folder and reads sampled SMILES."""
        work_folder = tempfile.mkdtemp()
        config_path = os.path.join(work_folder, self.CONFIG_FILE_NAME)
        output_path = os.path.join(work_folder, self.OUTPUT_FILE_NAME)

        config = {
            "run_type": "sampling",
            "device": self.DEVICE,
            "parameters": {
                "model_file": self.model_file,
                "num_smiles": self.NUM_SMILES,
                "output_file": output_path,
            },
        }
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file)

        subprocess.run([self.executable, config_path], check=True)

        with open(output_path, newline="", encoding="utf-8") as output_file:
            return [
                {"mol_id": f"REINV_{index}", "smiles": row["SMILES"]}
                for index, row in enumerate(csv.DictReader(output_file))
            ]
