import csv
import json
import os
import subprocess
import tempfile

from pipeline.sources.base import Source


class ReinventSource(Source):
    def get_rows(self) -> list[dict]:
        executable = os.environ["REINVENT_EXECUTABLE"]
        model_file = os.environ["REINVENT_MODEL_FILE"]

        work_folder = tempfile.mkdtemp()
        config_path = os.path.join(work_folder, "sampling.json")
        output_path = os.path.join(work_folder, "sampling.csv")

        config = {
            "run_type": "sampling",
            "device": "cpu",
            "parameters": {
                "model_file": model_file,
                "num_smiles": 15,
                "output_file": output_path
            },
        }
        with open(config_path, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file)

        result = subprocess.run([executable, config_path], capture_output=True, text=True)
        print("код завершения:", result.returncode)

        with open(output_path, newline="", encoding="utf-8") as output_file:
            return [
                {
                    "mol_id": f"REINV_{index}",
                    "smiles": row["SMILES"]
                }
                for index, row in enumerate(csv.DictReader(output_file))
            ]
