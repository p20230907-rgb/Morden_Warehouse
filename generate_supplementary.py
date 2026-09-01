#!/usr/bin/env python3
"""
generate_supplementary.py

Generates supplementary materials: prompt templates, query set, JSON schema, etc.
Uses the actual prompts extracted from newapp.py.
"""

import os
import json
import shutil
import sys
from run_full_ablation import SAMPLE_QUERIES  # Import the 60 queries

# Create output folder
os.makedirs("supplementary", exist_ok=True)

# 1. Save the 60-query set
with open("supplementary/query_set.json", "w") as f:
    json.dump(SAMPLE_QUERIES, f, indent=2)

# 2. Use the extracted prompt templates if available
if os.path.exists("prompt_templates.json"):
    shutil.copy("prompt_templates.json", "supplementary/prompt_templates.json")
    print("✅ Copied prompt_templates.json (extracted from newapp.py)")
else:
    print("⚠️ prompt_templates.json not found; run extract_prompts.py first.")

# 3. JSON schema for robot dispatch
dispatch_schema = {
    "type": "object",
    "properties": {
        "shelf_id": {"type": "string", "pattern": "^[A-G]$"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "qty": {"type": "integer", "minimum": 1}
                },
                "required": ["name", "qty"]
            }
        },
        "priority": {"type": "integer", "minimum": 1, "maximum": 5, "default": 3},
        "avoid_obstacles": {"type": "boolean", "default": True},
        "max_retries": {"type": "integer", "minimum": 1, "default": 3}
    },
    "required": ["shelf_id", "items"]
}
with open("supplementary/dispatch_schema.json", "w") as f:
    json.dump(dispatch_schema, f, indent=2)

# 4. Copy inventory CSV if it exists
if os.path.exists("shelfinfo.csv"):
    shutil.copy("shelfinfo.csv", "supplementary/shelfinfo.csv")
else:
    print("⚠️ shelfinfo.csv not found; skipping.")

# 5. Copy safety manual if it exists
if os.path.exists("WAREHOUSE_SAFETY_MANUAL.txt"):
    shutil.copy("WAREHOUSE_SAFETY_MANUAL.txt", "supplementary/WAREHOUSE_SAFETY_MANUAL.txt")
else:
    print("⚠️ WAREHOUSE_SAFETY_MANUAL.txt not found; skipping.")

# 6. Copy router training data if it exists
if os.path.exists("router_training.csv"):
    shutil.copy("router_training.csv", "supplementary/router_training.csv")

# 7. Create a README
readme = """
Supplementary Materials for "A Tiered Expert System for Evidence-Grounded Warehouse Assistance and Robot Dispatch"

- query_set.json: the 60 balanced queries used for evaluation.
- prompt_templates.json: the full prompts used for each LLM call (extracted from code).
- dispatch_schema.json: JSON schema for ROS 2 robot dispatch.
- shelfinfo.csv: sample inventory data.
- WAREHOUSE_SAFETY_MANUAL.txt: safety manual text (if available).
- router_training.csv: labelled data for training the classifier router (optional).

For full reproducibility, see the main code repository.
"""
with open("supplementary/README.txt", "w") as f:
    f.write(readme)

print("✅ Supplementary materials generated in the 'supplementary' folder.")