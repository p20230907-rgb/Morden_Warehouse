#!/usr/bin/env python3
"""
run_full_ablation.py – Extended ablation with exactly 60 distinct queries.
Runs against the running Flask app and appends results to full_ablation_results.csv.
"""

import os
import sys
import time
import json
import re
import requests
import pandas as pd
from datetime import datetime
from config import CONFIG 

# Override CONFIG from environment variables (if set)
if os.getenv("CHUNK_SIZE"):
    CONFIG["chunk_size"] = int(os.getenv("CHUNK_SIZE"))
if os.getenv("CHUNK_OVERLAP"):
    CONFIG["chunk_overlap"] = int(os.getenv("CHUNK_OVERLAP"))
if os.getenv("FUSION_WEIGHTS"):
    CONFIG["fusion_weights"] = json.loads(os.getenv("FUSION_WEIGHTS"))

FLASK_URL = "http://localhost:5000/query"

# ---------- 60 Distinct Queries ----------
SAMPLE_QUERIES = [
    # --- Direct Item Lookup (10 queries) ---
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

    # --- LLM Shelf Extraction (8 queries) ---
    {"query": "What's in Shelf A?", "mode": "auto", "type": "LLM Shelf Extraction"},
    {"query": "Show me items on Shelf D.", "mode": "auto", "type": "LLM Shelf Extraction"},
    {"query": "Navigate to Shelf F.", "mode": "auto", "type": "LLM Shelf Extraction"},
    {"query": "I need assistance near shelf B", "mode": "auto", "type": "LLM Shelf Extraction"},
    {"query": "Tell me about shelf G.", "mode": "auto", "type": "LLM Shelf Extraction"},
    {"query": "What's at C?", "mode": "auto", "type": "LLM Shelf Extraction"},
    {"query": "Items on Shelf E", "mode": "auto", "type": "LLM Shelf Extraction"},
    {"query": "Shelf A contents", "mode": "auto", "type": "LLM Shelf Extraction"},

    # --- RAG Safety (12 queries) ---
    {"query": "What PPE is required for handling chemicals?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What are the rules for emergency exits?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What is the LOTO procedure?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What's the replacement frequency for safety helmets?", "mode": "auto", "type": "RAG Safety"},
    {"query": "How often do we inspect fire extinguishers?", "mode": "auto", "type": "RAG Safety"},
    {"query": "Who can operate forklifts?", "mode": "auto", "type": "RAG Safety"},
    {"query": "Where should I store hazardous materials?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What kind of gloves do I need for sharp objects?", "mode": "auto", "type": "RAG Safety"},
    {"query": "Tell me about the evacuation protocol.", "mode": "auto", "type": "RAG Safety"},
    {"query": "What standard must steel-toed boots conform to?", "mode": "auto", "type": "RAG Safety"},
    {"query": "How often are fire drills conducted?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What is the procedure for chemical spills?", "mode": "auto", "type": "RAG Safety"},

    # --- Traditional Search (Exact) (10 queries) ---
    {"query": "Safety Helmets", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},
    {"query": "Fire Extinguishers", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},
    {"query": "Spill Containment", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},
    {"query": "forklift operators", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},
    {"query": "Annual requirement (every 12 months)", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},
    {"query": "Steel-Toed Boots ISO 20345", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},
    {"query": "Disposal Follow EPA Regulation CFR 261", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},
    {"query": "Shelf B", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},
    {"query": "Apple iPhone 14", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},
    {"query": "PPE requirements", "mode": "traditional_search_only", "type": "Traditional Search (Exact)"},

    # --- Traditional Search (Semantic Fail) – 6 queries ---
    {"query": "protective gloves for chemicals", "mode": "traditional_search_only", "type": "Traditional Search (Semantic Fail)"},
    {"query": "exits for fire emergencies", "mode": "traditional_search_only", "type": "Traditional Search (Semantic Fail)"},
    {"query": "lockout tagout procedure steps", "mode": "traditional_search_only", "type": "Traditional Search (Semantic Fail)"},
    {"query": "recertification for driving warehouse vehicles", "mode": "traditional_search_only", "type": "Traditional Search (Semantic Fail)"},
    {"query": "location of Shelf A", "mode": "traditional_search_only", "type": "Traditional Search (Semantic Fail)"},
    {"query": "how to dispose of chemicals", "mode": "traditional_search_only", "type": "Traditional Search (Semantic Fail)"},

    # --- RAG (Semantic Success) – 6 queries ---
    {"query": "protective gloves for chemicals", "mode": "rag_only", "type": "RAG (Semantic Success)"},
    {"query": "exits for fire emergencies", "mode": "rag_only", "type": "RAG (Semantic Success)"},
    {"query": "lockout tagout procedure steps", "mode": "rag_only", "type": "RAG (Semantic Success)"},
    {"query": "recertification for driving warehouse vehicles", "mode": "rag_only", "type": "RAG (Semantic Success)"},
    {"query": "location of Shelf A", "mode": "rag_only", "type": "RAG (Semantic Success)"},
    {"query": "how to dispose of chemicals", "mode": "rag_only", "type": "RAG (Semantic Success)"},

    # --- Mixed (4 queries) ---
    {"query": "How many Apple iPhone 14s are there and what PPE for general work?", "mode": "auto", "type": "Mixed"},
    {"query": "Give me details on Shelf B and chemical disposal rules.", "mode": "auto", "type": "Mixed"},
    {"query": "What's the stock of Dell XPS 13 and where are the assembly points?", "mode": "auto", "type": "Mixed"},
    {"query": "Tell me about Shelf C and fire extinguisher inspection.", "mode": "auto", "type": "Mixed"},

    # --- Failure (Not Found) – 4 queries ---
    {"query": "Where is the Quantum Teleporter?", "mode": "auto", "type": "Failure (Not Found)"},
    {"query": "Tell me about Shelf Z.", "mode": "auto", "type": "Failure (Not Found)"},
    {"query": "What are the rules for flying drones in the warehouse?", "mode": "auto", "type": "Failure (Not Found)"},
    {"query": "Who is the CEO?", "mode": "auto", "type": "Failure (Not Found)"},
]

# Verify we have exactly 60 queries
assert len(SAMPLE_QUERIES) == 60, f"Expected 60 queries, got {len(SAMPLE_QUERIES)}"

OUTPUT_FILE = "full_ablation_results.csv"

def check_correctness(message, query):
    """Simplified correctness check for the ablation."""
    if not message:
        return False
    msg_lower = message.lower()
    # If query is an item name, check if it appears in the message
    words = set(re.findall(r'\b\w+\b', query.lower()))
    stopwords = {"the", "a", "an", "of", "for", "on", "at", "to", "in", "with", "without", "but", "or", "and", "what", "is", "are", "do", "we", "have", "you", "i", "me", "my", "how", "many", "tell", "me", "about", "show", "items", "where", "who", "which", "when", "why"}
    words = words - stopwords
    if words and any(w in msg_lower for w in words):
        return True
    # Check for numbers in count queries
    if "how many" in query.lower() or "stock" in query.lower():
        if re.search(r'\b\d+\b', msg_lower):
            return True
    return False

def run_ablation():
    model = os.getenv("TOGETHER_MODEL", "unknown")
    print(f"Running full ablation for model: {model}")
    print(f"Testing {len(SAMPLE_QUERIES)} distinct queries...")

    results = []
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        query_text = q["query"]
        qtype = q["type"]
        mode = q.get("mode", "auto")
        print(f"  [{i}/{len(SAMPLE_QUERIES)}] {query_text} (mode: {mode})")
        start = time.time()
        try:
            resp = requests.post(FLASK_URL, json={"query": query_text, "mode": mode}, timeout=30)
            latency = time.time() - start
            data = resp.json()
            message = data.get("message", "")
            tier = data.get("tier", "unknown")
            confidence = data.get("confidence", 0.0)
            status = data.get("status", "error")
            correct = check_correctness(message, query_text) if status == "success" else False
            print(f"      -> tier={tier}, correct={correct}, latency={latency:.3f}s")
        except Exception as e:
            latency = time.time() - start
            message = f"Error: {str(e)}"
            tier = "error"
            confidence = 0.0
            correct = False
            print(f"      -> ERROR: {e}")

        results.append({
            "model": model,
            "query": query_text,
            "type": qtype,
            "mode": mode,
            "tier": tier,
            "confidence": confidence,
            "latency": latency,
            "correct": correct,
            "message": message[:200],
            "timestamp": datetime.now().isoformat()
        })

    df_new = pd.DataFrame(results)
    if os.path.exists(OUTPUT_FILE):
        df_existing = pd.read_csv(OUTPUT_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(OUTPUT_FILE, index=False)

    total = len(results)
    correct_count = sum(1 for r in results if r["correct"])
    print(f"✅ Results appended to {OUTPUT_FILE}")
    print(f"   Model: {model}, Queries: {total}, Correct: {correct_count}/{total} ({correct_count/total*100:.1f}%)")

if __name__ == "__main__":
    run_ablation()