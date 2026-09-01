#!/usr/bin/env python3
"""
latency_stats.py

Reads the combined ablation CSV (from manual verification) and computes
latency statistics per model and per tier.
"""

import pandas as pd
import numpy as np

# Input file (you can also use the raw ablation CSVs directly)
input_file = "manual_verification_all_models.csv"

# If you haven't combined yet, you can read the three original files
# and combine them here. For simplicity, we assume you have a single CSV.
# If you only have the raw files, you can use the combine script first.

df = pd.read_csv(input_file)

# Ensure latency is numeric
df['latency'] = pd.to_numeric(df['latency'], errors='coerce')

# Group by model and compute latency stats
latency_by_model = df.groupby('model')['latency'].agg(['mean', 'std', 'median', lambda x: x.quantile(0.95)])
latency_by_model.columns = ['mean', 'std', 'median', 'p95']
print("\n=== Latency per model ===")
print(latency_by_model.round(3))

# Group by model and tier
latency_by_model_tier = df.groupby(['model', 'tier'])['latency'].agg(['mean', 'std', 'median', lambda x: x.quantile(0.95)])
latency_by_model_tier.columns = ['mean', 'std', 'median', 'p95']
print("\n=== Latency per model and tier ===")
print(latency_by_model_tier.round(3))

# Save to CSV
latency_by_model.to_csv("latency_by_model.csv")
latency_by_model_tier.to_csv("latency_by_model_tier.csv")
print("\n✅ Latency statistics saved to CSV files.")