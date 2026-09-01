#!/usr/bin/env python3
"""
manual_verify_3tier.py – Manually verify each query response with 3-tier labels.
Outputs a CSV ready for analysis.
"""

import pandas as pd
import os
import sys
import numpy as np
from ground_truth_answers import GROUND_TRUTH

def classify_auto(message, ground):
    """Auto-classify as Correct, Partial, or Incorrect based on keywords."""
    # Handle NaN or None messages
    if pd.isna(message) or message is None:
        return "incorrect"
    
    # Ensure message is string
    msg_lower = str(message).lower()
    
    if not ground:
        return "incorrect"
    
    must_have = ground.get("must_have", [])
    should_have = ground.get("should_have", [])
    
    # Count matches
    must_matches = sum(1 for kw in must_have if kw.lower() in msg_lower)
    should_matches = sum(1 for kw in should_have if kw.lower() in msg_lower)
    
    # Decision logic
    if must_matches == len(must_have) and len(must_have) > 0:
        if should_matches >= len(should_have) * 0.5:
            return "correct"
        else:
            return "partial"
    elif must_matches >= len(must_have) * 0.5:
        return "partial"
    else:
        return "incorrect"


def generate_verification_file(input_csv, output_csv):
    """Generate a CSV for manual 3-tier verification."""
    df = pd.read_csv(input_csv)
    
    results = []
    for idx, row in df.iterrows():
        query = row['query']
        message = row.get('message', '')
        
        # Handle NaN or None message
        if pd.isna(message) or message is None:
            message_str = ""
        else:
            message_str = str(message)
        
        ground = GROUND_TRUTH.get(query, None)
        
        if ground is None:
            auto_label = "unknown"
            ground_text = "NOT DEFINED"
        else:
            auto_label = classify_auto(message_str, ground)
            ground_text = ground.get("expected", "")
        
        results.append({
            'query': query,
            'type': row.get('type', ''),
            'tier': row.get('tier', ''),
            'model': row.get('model', 'unknown'),
            'auto_label': auto_label,
            'manual_label': '',  # To be filled by user
            'notes': '',         # Optional notes
            'message': message_str,
            'ground_truth': ground_text,
        })
    
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv, index=False)
    print(f"✅ Verification file created: {output_csv}")
    print(f"   Total queries: {len(df_out)}")
    print(f"   Auto labels: {df_out['auto_label'].value_counts().to_dict()}")
    print(f"\n📝 Open {output_csv} and fill the 'manual_label' column with:")
    print("   - 'correct'   (fully correct)")
    print("   - 'partial'   (mostly correct, missing some detail)")
    print("   - 'incorrect' (wrong or missing key information)")


def combine_and_verify():
    """Generate verification files for all three models."""
    files = [
        ("full_ablation_results_20b_alltest_nobase.csv", "verify_20b_3tier.csv"),
        ("full_ablation_results_70b_alltest_nobase.csv", "verify_70b_3tier.csv"),
        ("full_ablation_results_120b_alltest_nobase.csv", "verify_120b_3tier.csv"),
    ]
    
    for input_file, output_file in files:
        if os.path.exists(input_file):
            print(f"\n{'='*60}")
            print(f"Processing: {input_file}")
            generate_verification_file(input_file, output_file)
        else:
            print(f"⚠️ File not found: {input_file}")


def analyze_manual_labels():
    """After manual labelling, compute statistics."""
    files = [
        "verify_20b_3tier.csv",
        "verify_70b_3tier.csv",
        "verify_120b_3tier.csv",
    ]
    
    all_results = []
    for f in files:
        if os.path.exists(f):
            df = pd.read_csv(f)
            # Ensure manual_label exists and has values
            if 'manual_label' not in df.columns:
                print(f"⚠️ {f} missing 'manual_label' column. Please add it.")
                continue
            # Filter out empty manual labels
            df_valid = df[df['manual_label'].notna() & (df['manual_label'] != '')]
            if len(df_valid) > 0:
                df_valid['model'] = f.replace('verify_', '').replace('_3tier.csv', '')
                all_results.append(df_valid)
            else:
                print(f"⚠️ No manual labels found in {f}. Please fill them first.")
    
    if not all_results:
        print("\n❌ No verification files with manual labels found.")
        print("Please run the script first to generate verification files, then fill the 'manual_label' column.")
        return
    
    combined = pd.concat(all_results, ignore_index=True)
    
    # Statistics per model
    model_stats = combined.groupby('model')['manual_label'].value_counts().unstack(fill_value=0)
    print("\n=== Manual Label Distribution per Model ===")
    print(model_stats)
    
    # Accuracy per model (Correct / Total)
    model_acc = combined.groupby('model').apply(
        lambda x: (x['manual_label'] == 'correct').sum() / len(x)
    ).round(3)
    print("\n=== Accuracy per Model ===")
    print(model_acc)
    
    # Accuracy per type
    type_acc = combined.groupby(['model', 'type']).apply(
        lambda x: (x['manual_label'] == 'correct').sum() / len(x)
    ).unstack().round(3)
    print("\n=== Accuracy per Model and Query Type ===")
    print(type_acc)
    
    # Compute Cohen's Kappa (if we have multiple annotators)
    # You'd need to add annotator columns for this
    # For now, just save the combined file
    combined.to_csv("verified_combined_3tier.csv", index=False)
    print("\n✅ Combined verification saved to: verified_combined_3tier.csv")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    print(f"Total manually verified queries: {len(combined)}")
    print(f"Correct: {len(combined[combined['manual_label'] == 'correct'])}")
    print(f"Partial: {len(combined[combined['manual_label'] == 'partial'])}")
    print(f"Incorrect: {len(combined[combined['manual_label'] == 'incorrect'])}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--analyze":
        analyze_manual_labels()
    else:
        combine_and_verify()
        print("\n" + "="*60)
        print("📝 NEXT STEPS:")
        print("1. Open each verify_*_3tier.csv file")
        print("2. Fill the 'manual_label' column (correct/partial/incorrect)")
        print("3. Add optional notes if needed")
        print("4. Run: python manual_verify_3tier.py --analyze")
        print("   to compute statistics from your labels")