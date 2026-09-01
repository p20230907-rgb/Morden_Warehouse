#!/usr/bin/env python3
"""
run_noisy_ablation.py

Generates noisy versions of the 60 queries and runs the ablation.
Outputs a CSV for manual verification (no auto-correct).
"""

import os
import sys
import json
import re
import time
import random
import requests
import pandas as pd
from datetime import datetime
from config import CONFIG

FLASK_URL = "http://localhost:5000/query"

# ---------- Import the original query list from run_full_ablation.py ----------
# For simplicity, we copy the balanced set here.
# You can also import from the original file.
SAMPLE_QUERIES = [
    # --- Direct (15) ---
    {"query": "Apple iPhone 14", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Samsung Galaxy S23", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Sony WH-1000XM5", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Dell XPS 13", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Bose QuietComfort 45", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Logitech MX Master 3", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Sony PlayStation 5", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Samsung Galaxy S24", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Apple iPhone 14 Pro", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Bose SoundLink Flex", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Shelf A", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Shelf B", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Shelf C", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Shelf D", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "How many Samsung Galaxy S23 are there?", "mode": "auto", "type": "Direct Item Lookup"},

    # --- Keyword (15) ---
    {"query": "Safety Helmets", "mode": "auto", "type": "Keyword Search"},
    {"query": "Fire Extinguishers", "mode": "auto", "type": "Keyword Search"},
    {"query": "forklift operators", "mode": "auto", "type": "Keyword Search"},
    {"query": "steel-toed boots", "mode": "auto", "type": "Keyword Search"},
    {"query": "cut-resistant gloves", "mode": "auto", "type": "Keyword Search"},
    {"query": "PPE requirements", "mode": "auto", "type": "Keyword Search"},
    {"query": "LOTO procedure", "mode": "auto", "type": "Keyword Search"},
    {"query": "emergency exits", "mode": "auto", "type": "Keyword Search"},
    {"query": "assembly points", "mode": "auto", "type": "Keyword Search"},
    {"query": "hazardous chemicals", "mode": "auto", "type": "Keyword Search"},
    {"query": "locked ventilated cabinets", "mode": "auto", "type": "Keyword Search"},
    {"query": "incident report", "mode": "auto", "type": "Keyword Search"},
    {"query": "first aid kit", "mode": "auto", "type": "Keyword Search"},
    {"query": "warehouse training", "mode": "auto", "type": "Keyword Search"},
    {"query": "spill containment", "mode": "auto", "type": "Keyword Search"},

    # --- RAG (15) ---
    {"query": "What PPE is required for handling chemicals?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What are the rules for emergency exits?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What is the LOTO procedure?", "mode": "auto", "type": "RAG Safety"},
    {"query": "How often do we inspect fire extinguishers?", "mode": "auto", "type": "RAG Safety"},
    {"query": "Who can operate forklifts?", "mode": "auto", "type": "RAG Safety"},
    {"query": "Where should I store hazardous materials?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What kind of gloves do I need for sharp objects?", "mode": "auto", "type": "RAG Safety"},
    {"query": "Tell me about the evacuation protocol.", "mode": "auto", "type": "RAG Safety"},
    {"query": "How often are fire drills conducted?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What is the procedure for chemical spills?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What standard must steel-toed boots conform to?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What's the replacement frequency for safety helmets?", "mode": "auto", "type": "RAG Safety"},
    {"query": "Explain the emergency evacuation procedure.", "mode": "auto", "type": "RAG Safety"},
    {"query": "What should I do during a warehouse fire?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What are the requirements for emergency exits?", "mode": "auto", "type": "RAG Safety"},

    # --- LLM (15) ---
    {"query": "Which iPhone model has the highest price?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Compare the stock on Shelf A and Shelf B.", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which shelf has the most items?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "What is the total value of all smartphones?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Compare the prices of Dell laptops.", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which item is the most expensive?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which shelf should I visit first to get the most expensive products?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "What is the difference between Shelf C and Shelf D?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Compare the inventory levels across shelves.", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which product has the lowest stock?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which shelf has the highest total inventory value?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Summarize the inventory on Shelf F.", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "What is the average stock per shelf?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which two shelves have similar stock levels?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Give me a short comparison of shelves A, B, and C.", "mode": "auto", "type": "LLM Reasoning"},
]


def insert_typos(text, p=0.1):
    """Insert random character swaps."""
    words = text.split()
    for i, w in enumerate(words):
        if len(w) > 2 and random.random() < p:
            idx = random.randint(0, len(w)-2)
            w = w[:idx] + w[idx+1] + w[idx] + w[idx+2:]
            words[i] = w
    return " ".join(words)

def generate_noisy_queries(queries, p=0.15):
    """Return a list of queries with typos."""
    noisy = []
    for q in queries:
        noisy_q = q.copy()
        noisy_q["query"] = insert_typos(q["query"], p)
        noisy.append(noisy_q)
    return noisy

def run_ablation():
    model = os.getenv("TOGETHER_MODEL", "unknown")
    print(f"Running noisy ablation for model: {model}")

    noisy_queries = generate_noisy_queries(SAMPLE_QUERIES, p=0.15)
    print(f"Generated {len(noisy_queries)} noisy queries.")

    results = []
    run_timestamp = datetime.now().isoformat()
    for i, q in enumerate(noisy_queries, 1):
        query_text = q["query"]
        qtype = q["type"]
        mode = q.get("mode", "auto")
        print(f"  [{i}/{len(noisy_queries)}] {query_text}")
        start = time.time()
        try:
            resp = requests.post(FLASK_URL, json={"query": query_text, "mode": mode}, timeout=30)
            latency = time.time() - start
            data = resp.json()
            message = data.get("message", "")
            tier = data.get("tier", "unknown")
            confidence = data.get("confidence", 0.0)
            status = data.get("status", "error")
            # Do NOT assign automatic correctness; leave blank for manual verification
            correct = ""  # Placeholder
            print(f"      -> tier={tier}, latency={latency:.3f}s")
        except Exception as e:
            latency = time.time() - start
            message = f"Error: {str(e)}"
            tier = "error"
            confidence = 0.0
            correct = ""
            print(f"      -> ERROR: {e}")

        results.append({
            "model": model,
            "query": query_text,
            "type": qtype,
            "mode": mode,
            "tier": tier,
            "confidence": confidence,
            "latency": latency,
            "correct": correct,  # will be filled manually
            "message": message[:200],
            "timestamp": run_timestamp,
            "chunk_size": CONFIG.get("chunk_size", "default"),
            "chunk_overlap": CONFIG.get("chunk_overlap", "default"),
            "fusion_weights": str(CONFIG.get("fusion_weights", "default")),
            "router_mode_env": os.getenv("ROUTER_MODE", "auto"),
        })

    df_new = pd.DataFrame(results)
    output_file = "noisy_ablation_results.csv"
    if os.path.exists(output_file):
        df_existing = pd.read_csv(output_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(output_file, index=False)
    print(f"✅ Results appended to {output_file}")
    print("   Please manually verify the answers and set 'correct' to TRUE or FALSE.")
    print("   You can use the manual_verification approach as before.")

if __name__ == "__main__":
    run_ablation()