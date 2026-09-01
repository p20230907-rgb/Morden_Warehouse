#!/usr/bin/env python3
"""
Compute Precision@k and Recall@k for RAG component for k = 1, 3, 5.
"""

import os
import json
import numpy as np
from config import CONFIG
from data_loader import load_data
from sentence_transformers import SentenceTransformer

# Load data and vector store
script_dir = os.path.dirname(os.path.abspath(__file__))
data = load_data(script_dir, CONFIG)
vector_store = data["vector_store"]
all_combined_texts = data["all_combined_texts"]

# Load ground truth relevant chunks
with open("ground_truth_relevant_chunks.json", "r") as f:
    ground_truth = json.load(f)

# Embedding model
embeddings = SentenceTransformer(CONFIG["embedding_model"])

def get_embedding(text):
    return embeddings.encode(text)

# List of k values to evaluate
k_values = [1, 3, 5]

print("Retrieval Metrics (Precision@k, Recall@k)\n")
print("-" * 60)

for k in k_values:
    precision_sum = 0.0
    recall_sum = 0.0
    num_queries = 0

    for query, relevant_indices in ground_truth.items():
        q_emb = get_embedding(query).reshape(1, -1)
        D, I = vector_store.index.search(q_emb, k)  # I is indices of top-k chunks
        retrieved_indices = I[0].tolist()
        retrieved_set = set(retrieved_indices)
        relevant_set = set(relevant_indices)
        tp = len(retrieved_set & relevant_set)
        precision = tp / k
        recall = tp / len(relevant_set) if relevant_set else 0.0
        precision_sum += precision
        recall_sum += recall
        num_queries += 1

    avg_precision = precision_sum / num_queries
    avg_recall = recall_sum / num_queries
    print(f"k={k}: Precision@{k} = {avg_precision:.3f}, Recall@{k} = {avg_recall:.3f}")

print("-" * 60)
print("Note: These metrics are averaged over the 48 safety queries.")