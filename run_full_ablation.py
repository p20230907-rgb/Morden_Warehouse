#!/usr/bin/env python3
"""
run_full_ablation.py – Extended ablation with exactly 60 distinct queries.
Uses a single timestamp for the entire run.
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

# ---- Override CONFIG from environment variables (if set) ----
if os.getenv("CHUNK_SIZE"):
    CONFIG["chunk_size"] = int(os.getenv("CHUNK_SIZE"))
if os.getenv("CHUNK_OVERLAP"):
    CONFIG["chunk_overlap"] = int(os.getenv("CHUNK_OVERLAP"))
if os.getenv("FUSION_WEIGHTS"):
    CONFIG["fusion_weights"] = json.loads(os.getenv("FUSION_WEIGHTS"))

FLASK_URL = "http://localhost:5000/query"

# ---------- 60 Unseen Queries (Balanced: 15 per tier) ----------
SAMPLE_QUERIES = [
    # --- Direct (15) ---
    {"query": "Apple iPhone 13", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Samsung Galaxy Tab S9", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Sony WH-CH720N", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Sony Extra Bass Headphones", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Dell Inspiron 14", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Dell Laptop Bag", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Bose QuietComfort Ultra", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Bose Soundbar 600", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Logitech G Pro X Superlight", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Logitech Webcam C920", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "PlayStation VR2", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "PlayStation 5 Games Bundle", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "PlayStation Charging Station", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Bose Replacement Ear Pads", "mode": "auto", "type": "Direct Item Lookup"},
    {"query": "Logitech Mouse Pad", "mode": "auto", "type": "Direct Item Lookup"},

    # --- Keyword (15) ---
    {"query": "orange reflective vest", "mode": "auto", "type": "Keyword Search"},
    {"query": "helmet use in loading zones", "mode": "auto", "type": "Keyword Search"},
    {"query": "forklift operator renewal", "mode": "auto", "type": "Keyword Search"},
    {"query": "chemical storage ventilation", "mode": "auto", "type": "Keyword Search"},
    {"query": "emergency exit markings", "mode": "auto", "type": "Keyword Search"},
    {"query": "first aid inspection", "mode": "auto", "type": "Keyword Search"},
    {"query": "warehouse incident portal", "mode": "auto", "type": "Keyword Search"},
    {"query": "LOTO annual review", "mode": "auto", "type": "Keyword Search"},
    {"query": "spill response pads", "mode": "auto", "type": "Keyword Search"},
    {"query": "visitor safety gear", "mode": "auto", "type": "Keyword Search"},
    {"query": "fire drill schedule", "mode": "auto", "type": "Keyword Search"},
    {"query": "hearing protection rule", "mode": "auto", "type": "Keyword Search"},
    {"query": "green waste bin", "mode": "auto", "type": "Keyword Search"},
    {"query": "damaged item inspection area", "mode": "auto", "type": "Keyword Search"},
    {"query": "inventory discrepancy reporting", "mode": "auto", "type": "Keyword Search"},

    # --- RAG (15) ---
    {"query": "What should a visitor wear before entering operational warehouse areas?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What are the documented requirements for a safe chemical-storage cabinet?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What must an operator do to remain qualified to drive a forklift?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What procedure is specified when an inventory discrepancy is discovered?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What should happen to damaged goods before further handling?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What are the manual's requirements for emergency assembly locations?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What reporting steps apply to an injury under the current 2025 procedure?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What does the manual require after a safety violation?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What are workers instructed to do when a spill is discovered?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What evacuation restrictions apply to elevators?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What inspection schedule applies to firefighting equipment?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What PPE is required when employees handle sharp objects?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What does the manual specify for hearing protection in noisy areas?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What are the requirements for the number and marking of emergency exits?", "mode": "auto", "type": "RAG Safety"},
    {"query": "What contact information is provided for a medical emergency?", "mode": "auto", "type": "RAG Safety"},

    # --- LLM (15) ---
    {"query": "By how many units does Shelf B exceed Shelf C?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which shelf is second highest in total stock?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Calculate the combined stock of Shelves A and G.", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which two products share the minimum stock level?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "How many total units are stored on Shelves D and E together?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Compare the inventory value of Dell XPS 13 with Dell XPS 15.", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "What is the combined stock of all Bose products?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which Samsung product contributes the most units to Shelf B?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Order Shelves D, E, and G from lowest to highest total stock.", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "What fraction of Shelf G stock is Sony PlayStation 5?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "How much inventory value is represented by the Sony PlayStation 5 stock?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Compare the combined stock of Apple phones with Samsung phones.", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Which shelf is closest to the warehouse-wide average stock per shelf?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "What is the combined stock of the three lowest-stock products?", "mode": "auto", "type": "LLM Reasoning"},
    {"query": "Give a short comparison of total stock on Shelves A, C, and F.", "mode": "auto", "type": "LLM Reasoning"},
]

# Verify we have exactly 60 queries
assert len(SAMPLE_QUERIES) == 60, f"Expected 60 queries, got {len(SAMPLE_QUERIES)}"

# ---- Override mode if FORCE_MODE environment variable is set ----
force_mode = os.getenv("FORCE_MODE")
if force_mode:
    for q in SAMPLE_QUERIES:
        q["mode"] = force_mode
    print(f"🔧 Forcing mode to: {force_mode} for all queries")
else:
    print("ℹ️ Using original modes from query definitions")

OUTPUT_FILE = "full_ablation_results_chunk_advanced_120b.csv"

def check_correctness(message, query):
    if not message:
        return False
    msg_lower = message.lower()
    words = set(re.findall(r'\b\w+\b', query.lower()))
    stopwords = {"the", "a", "an", "of", "for", "on", "at", "to", "in", "with", "without", "but", "or", "and", "what", "is", "are", "do", "we", "have", "you", "i", "me", "my", "how", "many", "tell", "me", "about", "show", "items", "where", "who", "which", "when", "why"}
    words = words - stopwords
    if words and any(w in msg_lower for w in words):
        return True
    if "how many" in query.lower() or "stock" in query.lower():
        if re.search(r'\b\d+\b', msg_lower):
            return True
    return False

def run_ablation():
    model = os.getenv("TOGETHER_MODEL", "unknown")
    chunk_size = CONFIG.get("chunk_size", "default")
    chunk_overlap = CONFIG.get("chunk_overlap", "default")
    fusion_weights = CONFIG.get("fusion_weights", "default")
    router_mode = os.getenv("ROUTER_MODE", "auto")
    actual_mode = SAMPLE_QUERIES[0]["mode"] if SAMPLE_QUERIES else "auto"

    print(f"Running full ablation for model: {model}")
    print(f"  chunk_size={chunk_size}, chunk_overlap={chunk_overlap}, fusion_weights={fusion_weights}")
    print(f"  router_mode_env={router_mode}, actual_mode_used={actual_mode}")
    print(f"Testing {len(SAMPLE_QUERIES)} distinct queries...")

    # --- Generate ONE timestamp for the entire run ---
    run_timestamp = datetime.now().isoformat()

    results = []
    for i, q in enumerate(SAMPLE_QUERIES, 1):
        query_text = q["query"]
        qtype = q["type"]
        mode = q["mode"]
        print(f"  [{i}/{len(SAMPLE_QUERIES)}] {query_text} (mode: {mode})")
        start = time.time()
        try:
            resp = requests.post(FLASK_URL, json={"query": query_text, "mode": mode}, timeout=120)
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
            "message": message,
            "timestamp": run_timestamp,    # same timestamp for all queries
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "fusion_weights": str(fusion_weights),
            "router_mode_env": router_mode,
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