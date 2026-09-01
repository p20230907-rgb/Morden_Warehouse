#!/usr/bin/env python3
"""
Analyse full_ablation_results.csv – automatically selects the most recent run
with at least 50 queries (i.e., a full ablation run).
"""

import sys
import pandas as pd
import json

# Load CSV
df = pd.read_csv("full_ablation_results.csv")

if df.empty:
    print("No data found in full_ablation_results.csv")
    sys.exit(1)

# Count rows per timestamp
ts_counts = df['timestamp'].value_counts()

# Filter runs with at least 50 queries (ignore short test runs)
valid_timestamps = ts_counts[ts_counts >= 50].index.tolist()

if not valid_timestamps:
    print("No run with at least 50 queries found. Please run the full ablation first.")
    print("Available runs:")
    for ts, count in ts_counts.items():
        print(f"  {ts} – {count} queries")
    sys.exit(1)

# Pick the most recent valid run
target_timestamp = max(valid_timestamps)
df_run = df[df['timestamp'] == target_timestamp].copy()

print(f"Selected run with {len(df_run)} queries at {target_timestamp}")

# Overall accuracy
overall_acc = df_run['correct'].mean()
print(f"\nOverall answer accuracy: {overall_acc*100:.1f}% ({df_run['correct'].sum()}/{len(df_run)})")

# Per-type accuracy
type_acc = df_run.groupby('type')['correct'].mean().sort_values(ascending=False)
print("\nAccuracy by query type:")
print(type_acc.round(3))

# Expected tier mapping (for balanced set)
EXPECTED_TIER_MAP = {
    "Direct Item Lookup": "direct",
    "Keyword Search": "keyword",
    "RAG Safety": "rag",
    "LLM Reasoning": "llm",
    "LLM Shelf Extraction": "llm",
    "Mixed": "rag",
    "Failure (Not Found)": "rag",
    "Traditional Search (Exact)": "keyword",
    "Traditional Search (Semantic Fail)": "keyword",
    "RAG (Semantic Success)": "rag",
}
df_run['expected_tier'] = df_run['type'].map(EXPECTED_TIER_MAP).fillna('unknown')

# Routing accuracy
df_run['routing_correct'] = df_run['tier'] == df_run['expected_tier']
routing_acc = df_run['routing_correct'].mean()
print(f"\nRouting accuracy (tier matches expected): {routing_acc*100:.1f}% ({df_run['routing_correct'].sum()}/{len(df_run)})")

# Confusion matrix
labels = sorted(df_run['expected_tier'].unique())
cm = pd.crosstab(df_run['expected_tier'], df_run['tier'], margins=True)
print("\nConfusion matrix (Expected vs Actual tier):")
print(cm)

# Misrouted queries
misrouted = df_run[~df_run['routing_correct']]
if len(misrouted) > 0:
    print(f"\nMisrouted queries ({len(misrouted)}):")
    for _, row in misrouted.iterrows():
        print(f"  {row['query']} (type: {row['type']}) → tier: {row['tier']}, expected: {row['expected_tier']}")
else:
    print("\nNo misrouted queries.")

# Tier-wise answer accuracy
tier_correct = df_run.groupby('tier')['correct'].mean()
print("\nAnswer accuracy by tier:")
print(tier_correct.round(3))

# Summary
summary = {
    "overall_accuracy": overall_acc,
    "routing_accuracy": routing_acc,
    "type_accuracy": type_acc.to_dict(),
    "tier_accuracy": tier_correct.to_dict(),
}
with open("analysis_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSummary saved to analysis_summary.json")