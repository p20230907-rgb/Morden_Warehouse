#!/usr/bin/env python3
"""
Run fusion-weight ablation.
Loops over weight configurations, sets environment variables, and runs run_full_ablation.py.
"""

import os
import json
import subprocess

weight_sets = [
    {"semantic": 0.6, "keyword": 0.3, "metadata": 0.1},   # current
    {"semantic": 0.7, "keyword": 0.2, "metadata": 0.1},
    {"semantic": 0.5, "keyword": 0.4, "metadata": 0.1},
    {"semantic": 0.4, "keyword": 0.3, "metadata": 0.3},
]

for weights in weight_sets:
    print(f"\n--- Running with weights: {weights} ---")
    os.environ["FUSION_WEIGHTS"] = json.dumps(weights)
    # Clear any chunk overrides
    os.environ.pop("CHUNK_SIZE", None)
    os.environ.pop("CHUNK_OVERLAP", None)
    subprocess.run(["python", "run_full_ablation.py"])

print("Weight ablation finished. Results in full_ablation_results.csv (appended).")