#!/usr/bin/env python3
"""
test_real_router.py – Test the actual rule‑based router (from router.py)
on the 60 ablation queries and report routing accuracy.
"""

import os
import sys
import numpy as np
import pandas as pd

# Import the real Router and SAFETY_KEYWORDS from your module
from router import Router, SAFETY_KEYWORDS
from data_loader import load_data
from config import CONFIG

# ----------------------------------------------------------------------
# Load shelf_locations (needed by the router)
# ----------------------------------------------------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
data = load_data(script_dir, CONFIG)
shelf_locations = data["shelf_locations"]

# Instantiate the router in rule‑based mode (or "confidence")
router = Router(mode="rule_based")  # "confidence" is also mapped to "rule_based"

# ----------------------------------------------------------------------
# 60 queries (exactly as in run_full_ablation.py)
# ----------------------------------------------------------------------
SAMPLE_QUERIES = [
    # --- Direct Item Lookup (10 queries) ---
    {"query": "Apple iPhone 14", "type": "Direct Item Lookup"},
    {"query": "Samsung Galaxy S23", "type": "Direct Item Lookup"},
    {"query": "Sony WH-1000XM5", "type": "Direct Item Lookup"},
    {"query": "Dell XPS 13", "type": "Direct Item Lookup"},
    {"query": "Bose QuietComfort 45", "type": "Direct Item Lookup"},
    {"query": "Logitech MX Master 3", "type": "Direct Item Lookup"},
    {"query": "Sony PlayStation 5", "type": "Direct Item Lookup"},
    {"query": "Samsung Galaxy S24", "type": "Direct Item Lookup"},
    {"query": "Apple iPhone 14 Pro", "type": "Direct Item Lookup"},
    {"query": "Bose SoundLink Flex", "type": "Direct Item Lookup"},

    # --- LLM Shelf Extraction (8 queries) ---
    {"query": "What's in Shelf A?", "type": "LLM Shelf Extraction"},
    {"query": "Show me items on Shelf D.", "type": "LLM Shelf Extraction"},
    {"query": "Navigate to Shelf F.", "type": "LLM Shelf Extraction"},
    {"query": "I need assistance near shelf B", "type": "LLM Shelf Extraction"},
    {"query": "Tell me about shelf G.", "type": "LLM Shelf Extraction"},
    {"query": "What's at C?", "type": "LLM Shelf Extraction"},
    {"query": "Items on Shelf E", "type": "LLM Shelf Extraction"},
    {"query": "Shelf A contents", "type": "LLM Shelf Extraction"},

    # --- RAG Safety (12 queries) ---
    {"query": "What PPE is required for handling chemicals?", "type": "RAG Safety"},
    {"query": "What are the rules for emergency exits?", "type": "RAG Safety"},
    {"query": "What is the LOTO procedure?", "type": "RAG Safety"},
    {"query": "What's the replacement frequency for safety helmets?", "type": "RAG Safety"},
    {"query": "How often do we inspect fire extinguishers?", "type": "RAG Safety"},
    {"query": "Who can operate forklifts?", "type": "RAG Safety"},
    {"query": "Where should I store hazardous materials?", "type": "RAG Safety"},
    {"query": "What kind of gloves do I need for sharp objects?", "type": "RAG Safety"},
    {"query": "Tell me about the evacuation protocol.", "type": "RAG Safety"},
    {"query": "What standard must steel-toed boots conform to?", "type": "RAG Safety"},
    {"query": "How often are fire drills conducted?", "type": "RAG Safety"},
    {"query": "What is the procedure for chemical spills?", "type": "RAG Safety"},

    # --- Traditional Search (Exact) (10 queries) ---
    {"query": "Safety Helmets", "type": "Traditional Search (Exact)"},
    {"query": "Fire Extinguishers", "type": "Traditional Search (Exact)"},
    {"query": "Spill Containment", "type": "Traditional Search (Exact)"},
    {"query": "forklift operators", "type": "Traditional Search (Exact)"},
    {"query": "Annual requirement (every 12 months)", "type": "Traditional Search (Exact)"},
    {"query": "Steel-Toed Boots ISO 20345", "type": "Traditional Search (Exact)"},
    {"query": "Disposal Follow EPA Regulation CFR 261", "type": "Traditional Search (Exact)"},
    {"query": "Shelf B", "type": "Traditional Search (Exact)"},
    {"query": "Apple iPhone 14", "type": "Traditional Search (Exact)"},
    {"query": "PPE requirements", "type": "Traditional Search (Exact)"},

    # --- Traditional Search (Semantic Fail) – 6 queries ---
    {"query": "protective gloves for chemicals", "type": "Traditional Search (Semantic Fail)"},
    {"query": "exits for fire emergencies", "type": "Traditional Search (Semantic Fail)"},
    {"query": "lockout tagout procedure steps", "type": "Traditional Search (Semantic Fail)"},
    {"query": "recertification for driving warehouse vehicles", "type": "Traditional Search (Semantic Fail)"},
    {"query": "location of Shelf A", "type": "Traditional Search (Semantic Fail)"},
    {"query": "how to dispose of chemicals", "type": "Traditional Search (Semantic Fail)"},

    # --- RAG (Semantic Success) – 6 queries ---
    {"query": "protective gloves for chemicals", "type": "RAG (Semantic Success)"},
    {"query": "exits for fire emergencies", "type": "RAG (Semantic Success)"},
    {"query": "lockout tagout procedure steps", "type": "RAG (Semantic Success)"},
    {"query": "recertification for driving warehouse vehicles", "type": "RAG (Semantic Success)"},
    {"query": "location of Shelf A", "type": "RAG (Semantic Success)"},
    {"query": "how to dispose of chemicals", "type": "RAG (Semantic Success)"},

    # --- Mixed (4 queries) ---
    {"query": "How many Apple iPhone 14s are there and what PPE for general work?", "type": "Mixed"},
    {"query": "Give me details on Shelf B and chemical disposal rules.", "type": "Mixed"},
    {"query": "What's the stock of Dell XPS 13 and where are the assembly points?", "type": "Mixed"},
    {"query": "Tell me about Shelf C and fire extinguisher inspection.", "type": "Mixed"},

    # --- Failure (Not Found) – 4 queries ---
    {"query": "Where is the Quantum Teleporter?", "type": "Failure (Not Found)"},
    {"query": "Tell me about Shelf Z.", "type": "Failure (Not Found)"},
    {"query": "What are the rules for flying drones in the warehouse?", "type": "Failure (Not Found)"},
    {"query": "Who is the CEO?", "type": "Failure (Not Found)"},
]

# ----------------------------------------------------------------------
# Expected tier mapping (based on query type)
# ----------------------------------------------------------------------
EXPECTED_TIER_MAP = {
    "Direct Item Lookup": "direct",
    "LLM Shelf Extraction": "llm",
    "RAG Safety": "rag",
    "Traditional Search (Exact)": "keyword",
    "Traditional Search (Semantic Fail)": "keyword",
    "RAG (Semantic Success)": "rag",
    "Mixed": "rag",
    "Failure (Not Found)": "rag",
}

# Prepare query list and expected tiers
queries = [q["query"] for q in SAMPLE_QUERIES]
expected = [EXPECTED_TIER_MAP.get(q["type"], "unknown") for q in SAMPLE_QUERIES]

# ----------------------------------------------------------------------
# Predictor using the real router
# ----------------------------------------------------------------------
def real_router_predict(query):
    tier, _ = router.predict(query, shelf_locations, SAFETY_KEYWORDS)
    return tier

# ----------------------------------------------------------------------
# Evaluation function
# ----------------------------------------------------------------------
def evaluate(predictor, name):
    preds = [predictor(q) for q in queries]
    correct = sum(1 for p, e in zip(preds, expected) if p == e)
    acc = correct / len(queries) * 100
    print(f"{name} accuracy: {acc:.1f}% ({correct}/{len(queries)})")
    # Confusion matrix
    labels = sorted(set(expected))
    cm = {l: {l2: 0 for l2 in labels} for l in labels}
    for p, e in zip(preds, expected):
        cm[e][p] += 1
    print(f"Confusion matrix ({name}):")
    print("         " + " ".join(labels))
    for e in labels:
        row = [str(cm[e][p]) for p in labels]
        print(f"{e:8} " + " ".join(row))
    return acc

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    print(f"Testing on {len(queries)} queries.")
    print("Expected tier distribution:", pd.Series(expected).value_counts().to_dict())
    evaluate(real_router_predict, "Real Rule-based Router")

if __name__ == "__main__":
    main()