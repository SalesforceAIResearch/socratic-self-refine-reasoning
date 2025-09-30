import json
import os
from functools import lru_cache


@lru_cache(maxsize=None)
def load_data(data_name, split="test", cot_path=None):
    # Determine file path
    file_path = f"experiment/data/{data_name}/{split}.json"

    # Load data from JSON file
    if os.path.exists(file_path):
        print(file_path)
        with open(file_path, 'r') as f:
            data = json.load(f)
    else:
        raise FileNotFoundError(f"Could not load {data_name} {split} locally")

    if cot_path is not None:
        if os.path.exists(cot_path):
            with open(cot_path, 'r') as f:
                cot_data = json.load(f)
            assert len(data) == len(cot_data)
            for entry, cot_entry in zip(data, cot_data):
                entry["cot_response"] = cot_entry["log"]["0"]["self_consistency_results"][0]["response"]
    else:
        for entry in data:
            entry["cot_response"] = None

    return data
