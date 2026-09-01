#!/usr/bin/env python3

import os
import json
import faiss
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from config import CONFIG
from data_loader import load_data


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

GROUND_TRUTH_FILE = os.path.join(
    SCRIPT_DIR,
    "ground_truth_relevant_chunks.json"
)

OUTPUT_FILE = os.path.join(
    SCRIPT_DIR,
    "embedding_retrieval_results.csv"
)


MODELS = [
    "paraphrase-MiniLM-L6-v2",
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
]

K_VALUES = [1, 3, 5]


# ==========================================================
# Load SAME chunks for every embedding
# ==========================================================

data = load_data(SCRIPT_DIR, CONFIG)

all_combined_texts = data["all_combined_texts"]

chunk_texts = []

for x in all_combined_texts:

    if isinstance(x, str):
        chunk_texts.append(x)

    elif hasattr(x, "page_content"):
        chunk_texts.append(x.page_content)

    else:
        chunk_texts.append(str(x))


print("Number of chunks:", len(chunk_texts))


# ==========================================================
# Ground truth
# ==========================================================

with open(
    GROUND_TRUTH_FILE,
    "r",
    encoding="utf-8"
) as f:

    ground_truth = json.load(f)


print(
    "Number of retrieval queries:",
    len(ground_truth)
)


results = []


# ==========================================================
# Test each embedding
# ==========================================================

for model_name in MODELS:

    print("\n" + "=" * 70)

    print("Embedding:", model_name)

    print("=" * 70)


    model = SentenceTransformer(model_name)


    # ------------------------------------------------------
    # Document embeddings
    # ------------------------------------------------------

    doc_embeddings = model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    doc_embeddings = np.asarray(
        doc_embeddings,
        dtype="float32"
    )


    # ------------------------------------------------------
    # Fresh FAISS index for THIS embedding
    # ------------------------------------------------------

    dimension = doc_embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(doc_embeddings)


    queries = list(
        ground_truth.keys()
    )


    query_embeddings = model.encode(
        queries,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    query_embeddings = np.asarray(
        query_embeddings,
        dtype="float32"
    )


    _, retrieved_indices = index.search(
        query_embeddings,
        max(K_VALUES)
    )


    result = {
        "embedding_model": model_name
    }


    for k in K_VALUES:

        precision_scores = []
        recall_scores = []


        for query_index, query in enumerate(queries):

            relevant = set(
                int(i)
                for i in ground_truth[query]
            )

            retrieved = set(
                int(i)
                for i in retrieved_indices[
                    query_index
                ][:k]
                if int(i) >= 0
            )


            true_positive = len(
                relevant & retrieved
            )


            precision = (
                true_positive / k
            )


            recall = (
                true_positive / len(relevant)
                if relevant
                else 0
            )


            precision_scores.append(
                precision
            )

            recall_scores.append(
                recall
            )


        result[f"P@{k}"] = np.mean(
            precision_scores
        )

        result[f"R@{k}"] = np.mean(
            recall_scores
        )


    results.append(result)


# ==========================================================
# Save
# ==========================================================

df = pd.DataFrame(results)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFINAL RESULTS")

print(df.to_string(index=False))

print("\nSaved:", OUTPUT_FILE)