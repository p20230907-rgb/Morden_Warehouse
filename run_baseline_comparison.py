#!/usr/bin/env python3
"""
Run baseline comparisons: auto, llm_only, rag_only, traditional_search_only, max_confidence.
"""

import os
import subprocess

baselines = [
    ("auto", None),
    ("direct_only", None),
    ("traditional_search_only", None),
    ("llm_only", None),
    ("rag_only", None),
]

for force_mode, router_mode in baselines:
    print("\n" + "="*60)
    print(f"Running baseline: force_mode={force_mode}, router_mode={router_mode}")
    print("="*60)

    if force_mode:
        os.environ["FORCE_MODE"] = force_mode
    else:
        os.environ.pop("FORCE_MODE", None)

    if router_mode:
        os.environ["ROUTER_MODE"] = router_mode
    else:
        os.environ.pop("ROUTER_MODE", None)

    subprocess.run(["python3", "run_full_ablation.py"])

print("\nAll baselines completed. Results are in full_ablation_results.csv")