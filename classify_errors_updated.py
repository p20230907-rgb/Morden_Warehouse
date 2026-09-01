#!/usr/bin/env python3
"""
classify_errors_updated.py

Classifies failures into:
  - data-resolution: direct lookup failed.
  - retrieval: keyword/rag returned no context or wrong context.
  - generation: llm reasoning failed, or malformed output.
  - routing: the tier chosen was incorrect (but this is covered by misrouting analysis).
  - unknown: otherwise.

Requires manual_correct column and the original query type.
"""

import pandas as pd
import re

def classify_error(row):
    # If answer is correct, no error
    if row['manual_correct']:
        return 'none'

    tier = row['tier']
    message = str(row['message']).lower()

    # Direct lookup failure
    if tier == 'direct':
        if 'item not found' in message:
            return 'data-resolution'
        else:
            return 'data-resolution'  # default

    # Keyword search failure
    if tier == 'keyword':
        if 'no keyword match found' in message or 'no relevant information' in message:
            return 'retrieval'
        else:
            return 'retrieval'  # assume retrieval issue

    # RAG failure
    if tier == 'rag':
        if 'no relevant context found' in message:
            return 'retrieval'
        elif 'i don\'t have enough information' in message or 'not in the context' in message:
            return 'generation'
        else:
            # Could be generation or retrieval; default to generation
            return 'generation'

    # LLM failure
    if tier == 'llm':
        if 'i\'m sorry' in message or 'i don\'t know' in message or 'not have information' in message:
            return 'generation'
        else:
            return 'generation'  # assume generation issue

    return 'unknown'

# Read the manually verified CSV
df = pd.read_csv("manual_verification_all_models.csv")

# Ensure manual_correct is boolean
def parse_manual(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().upper() == "TRUE"
    return False
df['manual_correct'] = df['manual_correct'].apply(parse_manual)

# Apply classification
df['error_type'] = df.apply(classify_error, axis=1)

# Count per model and error type
error_counts = df.groupby(['model', 'error_type']).size().unstack(fill_value=0)
print("\n=== Error counts per model ===")
print(error_counts)

# Percentages
total_per_model = df.groupby('model').size()
error_percent = error_counts.div(total_per_model, axis=0) * 100
print("\n=== Error percentages per model ===")
print(error_percent.round(1))

# Save to Excel
with pd.ExcelWriter("error_analysis_updated.xlsx") as writer:
    error_counts.to_excel(writer, sheet_name='Counts')
    error_percent.to_excel(writer, sheet_name='Percentages')

print("\n✅ Updated error analysis saved to error_analysis_updated.xlsx")