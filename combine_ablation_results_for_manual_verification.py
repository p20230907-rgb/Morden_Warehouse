#!/usr/bin/env python3
"""
combine_ablation_results_for_manual_verification.py

Reads the three ablation CSV files (20B, 70B, 120B) and creates a
single CSV for manual verification of answer correctness.
"""

import pandas as pd
import glob
import os

# Define the file paths (adjust if needed)
file_map = {
    "20B": "full_ablation_results_20b_alltest.csv",
    "70B": "full_ablation_results_70b_alltest.csv",
    "120B": "full_ablation_results_120b_alltest.csv",
}

# Output file
output_file = "manual_verification_all_models.csv"

# Columns we want to keep and reorder
keep_cols = ["model", "query", "type", "tier", "correct", "latency", "message"]
rename_cols = {
    "correct": "auto_correct"
}

all_dfs = []
for model_name, filename in file_map.items():
    if not os.path.exists(filename):
        print(f"⚠️ File not found: {filename} – skipping")
        continue
    df = pd.read_csv(filename)
    # Add model column
    df["model"] = model_name
    # Select and rename columns
    df_subset = df[keep_cols].copy()
    df_subset.rename(columns=rename_cols, inplace=True)
    # Add empty columns for manual verification
    df_subset["manual_correct"] = ""
    df_subset["notes"] = ""
    all_dfs.append(df_subset)

if not all_dfs:
    print("❌ No files found. Please check the file paths.")
    exit(1)

# Concatenate all runs
combined = pd.concat(all_dfs, ignore_index=True)

# Reorder columns for convenience
column_order = ["model", "query", "type", "tier", "auto_correct", "manual_correct", "message", "notes", "latency"]
combined = combined[column_order]

# Write to CSV
combined.to_csv(output_file, index=False)
print(f"✅ Created {output_file} with {len(combined)} rows for manual verification.")
print("   Open it in your spreadsheet editor, read each answer, and set 'manual_correct' to TRUE or FALSE.")