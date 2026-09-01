#!/usr/bin/env python3
"""
real_hybrid_fusion_test.py

Implements a REAL semantic + keyword retrieval fusion experiment.

What it does
------------
1. Loads your warehouse corpus and FAISS index using existing data_loader.py.
2. Computes:
   - normalized semantic similarity score from FAISS distances
   - normalized keyword-overlap score
3. Combines them using configurable weights:
      final_score = w_sem * semantic_score + w_kw * keyword_score
4. Tests multiple configurations:
   - semantic-only
   - keyword-only
   - 0.5/0.5
   - 0.6/0.4
   - 0.7/0.3
   - 0.8/0.2
5. Retrieves top-k passages for each query.
6. Optionally calls your Flask /query endpoint in RAG mode using the selected context
   ONLY if you adapt the endpoint. By default, this script evaluates retrieval only.

Outputs
-------
hybrid_fusion_retrieval_results.csv
hybrid_fusion_summary.csv

Recommended use for the paper
-----------------------------
Use retrieval metrics first. If you want end-to-end answer accuracy per fusion setting,
I provide a companion script below that calls a dedicated RAG endpoint.
"""

import csv
import math
import re
from pathlib import Path
from collections import defaultdict

import numpy as np

from config import CONFIG
from data_loader import load_data


# ------------------------------------------------------------
# CONFIGURATIONS TO TEST
# ------------------------------------------------------------

WEIGHT_CONFIGS = {
    "semantic_only": {"semantic": 1.0, "keyword": 0.0},
    "keyword_only": {"semantic": 0.0, "keyword": 1.0},
    "hybrid_50_50": {"semantic": 0.5, "keyword": 0.5},
    "hybrid_60_40": {"semantic": 0.6, "keyword": 0.4},
    "hybrid_70_30": {"semantic": 0.7, "keyword": 0.3},
    "hybrid_80_20": {"semantic": 0.8, "keyword": 0.2},
}

TOP_K = 5

# Use the same 60-query ground-truth file if present.
QUERY_FILE = "router_test_unseen_60_with_ground_truth_final.csv"

# If you have a retrieval-relevance file with relevant chunk IDs/texts,
# point this to it. Otherwise the script still runs and reports selections.
RELEVANCE_FILE = None


# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", str(text).lower())


def keyword_raw_score(query, text):
    """
    Token-overlap score normalized by query token count.
    Range: [0, 1]
    """
    q = set(tokenize(query))
    d = set(tokenize(text))
    if not q:
        return 0.0
    return len(q.intersection(d)) / len(q)


def semantic_similarity_from_l2(distance):
    """
    Convert FAISS squared-L2 distance to a bounded similarity-like score.

    similarity = 1 / (1 + distance)

    This preserves ranking and yields values in (0,1].
    """
    if distance is None or math.isinf(distance):
        return 0.0
    return 1.0 / (1.0 + float(distance))


def get_query_embedding(vector_store, query):
    ef = vector_store.embedding_function

    if hasattr(ef, "embed_query"):
        vec = ef.embed_query(query)
    elif callable(ef):
        vec = ef(query)
    else:
        raise RuntimeError(
            "vector_store.embedding_function cannot create query embeddings"
        )

    return np.array([vec], dtype="float32")


def get_all_documents(vector_store):
    """
    Returns docs in FAISS index order.
    """
    docs = []
    for idx in range(vector_store.index.ntotal):
        doc_id = vector_store.index_to_docstore_id[idx]
        doc = vector_store.docstore.search(doc_id)
        text = getattr(doc, "page_content", str(doc))
        docs.append(text)
    return docs


def semantic_scores_for_all_docs(query, vector_store):
    """
    Query FAISS for ALL documents so semantic and keyword scores
    are computed over the same candidate set.
    """
    qvec = get_query_embedding(vector_store, query)
    n = vector_store.index.ntotal

    D, I = vector_store.index.search(qvec, n)

    scores = {}
    distances = {}

    for dist, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        scores[int(idx)] = semantic_similarity_from_l2(float(dist))
        distances[int(idx)] = float(dist)

    return scores, distances


def minmax_normalize(score_dict):
    """
    Optional per-query normalization to [0,1].
    """
    if not score_dict:
        return {}

    vals = list(score_dict.values())
    mn, mx = min(vals), max(vals)

    if abs(mx - mn) < 1e-12:
        return {k: 1.0 for k in score_dict}

    return {k: (v - mn) / (mx - mn) for k, v in score_dict.items()}


def load_queries(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def load_relevance(path):
    """
    Optional relevance file format:
        query,relevant_text

    Multiple rows per query are allowed.
    """
    rel = defaultdict(list)
    if not path or not Path(path).exists():
        return rel

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r.get("query", "")
            t = r.get("relevant_text", "")
            if q and t:
                rel[q].append(t)
    return rel


def is_relevant(doc_text, relevant_texts):
    """
    Conservative text containment match.
    Replace this with your existing relevance annotation logic if available.
    """
    d = " ".join(str(doc_text).lower().split())

    for r in relevant_texts:
        rr = " ".join(str(r).lower().split())
        if rr and (rr in d or d in rr):
            return True

    return False


# ------------------------------------------------------------
# MAIN EXPERIMENT
# ------------------------------------------------------------

def main():
    base = Path(__file__).resolve().parent
    query_path = base / QUERY_FILE

    if not query_path.exists():
        raise FileNotFoundError(
            f"Missing query file: {query_path}\n"
            "Copy router_test_unseen_60_with_ground_truth_final.csv "
            "into the same folder."
        )

    print("Loading warehouse data...")
    data = load_data(str(base), CONFIG)

    vector_store = data["vector_store"]

    if vector_store is None:
        raise RuntimeError("Vector store is None.")

    docs = get_all_documents(vector_store)
    queries = load_queries(query_path)
    relevance = load_relevance(
        str(base / RELEVANCE_FILE) if RELEVANCE_FILE else None
    )

    print(f"Loaded {len(docs)} indexed passages.")
    print(f"Loaded {len(queries)} queries.")
    print(f"Testing {len(WEIGHT_CONFIGS)} fusion configurations.\n")

    detailed_rows = []
    summary = defaultdict(lambda: {
        "queries": 0,
        "p1_sum": 0.0,
        "r1_sum": 0.0,
        "p3_sum": 0.0,
        "r3_sum": 0.0,
        "p5_sum": 0.0,
        "r5_sum": 0.0,
        "top1_texts": []
    })

    for qi, row in enumerate(queries, start=1):
        query = row["query"]

        print(f"[{qi}/{len(queries)}] {query}")

        sem_raw, distances = semantic_scores_for_all_docs(
            query, vector_store
        )

        kw_raw = {
            idx: keyword_raw_score(query, docs[idx])
            for idx in range(len(docs))
        }

        # Normalize both score families per query to [0,1].
        sem = minmax_normalize(sem_raw)
        kw = minmax_normalize(kw_raw)

        relevant_texts = relevance.get(query, [])

        for config_name, w in WEIGHT_CONFIGS.items():
            fused = {}

            for idx in range(len(docs)):
                fused[idx] = (
                    w["semantic"] * sem.get(idx, 0.0)
                    + w["keyword"] * kw.get(idx, 0.0)
                )

            ranked = sorted(
                fused.items(),
                key=lambda x: x[1],
                reverse=True
            )

            top = ranked[:TOP_K]

            top_texts = [docs[idx] for idx, _ in top]

            # Metrics are calculated only when relevance annotations exist.
            metrics = {}
            for k in [1, 3, 5]:
                selected = top_texts[:k]

                if relevant_texts:
                    rel_flags = [
                        is_relevant(t, relevant_texts)
                        for t in selected
                    ]

                    precision = sum(rel_flags) / k

                    recovered = 0
                    for rel_text in relevant_texts:
                        if any(
                            is_relevant(t, [rel_text])
                            for t in selected
                        ):
                            recovered += 1

                    recall = (
                        recovered / len(relevant_texts)
                        if relevant_texts else 0.0
                    )
                else:
                    precision = float("nan")
                    recall = float("nan")

                metrics[k] = (precision, recall)

            s = summary[config_name]
            s["queries"] += 1
            s["top1_texts"].append(top_texts[0] if top_texts else "")

            if relevant_texts:
                s["p1_sum"] += metrics[1][0]
                s["r1_sum"] += metrics[1][1]
                s["p3_sum"] += metrics[3][0]
                s["r3_sum"] += metrics[3][1]
                s["p5_sum"] += metrics[5][0]
                s["r5_sum"] += metrics[5][1]

            for rank, (idx, fused_score) in enumerate(top, start=1):
                detailed_rows.append({
                    "query": query,
                    "expected_tier": row.get("expected_tier", ""),
                    "correct_answer": row.get("correct_answer", ""),
                    "config": config_name,
                    "semantic_weight": w["semantic"],
                    "keyword_weight": w["keyword"],
                    "rank": rank,
                    "doc_index": idx,
                    "semantic_score": sem.get(idx, 0.0),
                    "keyword_score": kw.get(idx, 0.0),
                    "fused_score": fused_score,
                    "faiss_distance": distances.get(idx, ""),
                    "retrieved_text": docs[idx],
                    "has_relevance_annotations": bool(relevant_texts),
                    "p_at_1": metrics[1][0],
                    "r_at_1": metrics[1][1],
                    "p_at_3": metrics[3][0],
                    "r_at_3": metrics[3][1],
                    "p_at_5": metrics[5][0],
                    "r_at_5": metrics[5][1],
                })

    detailed_out = base / "hybrid_fusion_retrieval_results.csv"

    with open(
        detailed_out,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=detailed_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(detailed_rows)

    summary_rows = []

    for config_name, s in summary.items():
        n = s["queries"]

        has_rel = any(
            r["config"] == config_name
            and r["has_relevance_annotations"]
            for r in detailed_rows
        )

        summary_rows.append({
            "config": config_name,
            "semantic_weight": WEIGHT_CONFIGS[config_name]["semantic"],
            "keyword_weight": WEIGHT_CONFIGS[config_name]["keyword"],
            "num_queries": n,
            "P@1": (
                s["p1_sum"] / n if has_rel else ""
            ),
            "R@1": (
                s["r1_sum"] / n if has_rel else ""
            ),
            "P@3": (
                s["p3_sum"] / n if has_rel else ""
            ),
            "R@3": (
                s["r3_sum"] / n if has_rel else ""
            ),
            "P@5": (
                s["p5_sum"] / n if has_rel else ""
            ),
            "R@5": (
                s["r5_sum"] / n if has_rel else ""
            ),
        })

    summary_out = base / "hybrid_fusion_summary.csv"

    with open(
        summary_out,
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=summary_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\nDONE")
    print(f"Detailed results: {detailed_out}")
    print(f"Summary results : {summary_out}")

    print("\nTOP-1 CHANGE CHECK")
    config_names = list(WEIGHT_CONFIGS.keys())

    for q in [r["query"] for r in queries]:
        top1 = []

        for cfg in config_names:
            matches = [
                r for r in detailed_rows
                if r["query"] == q
                and r["config"] == cfg
                and r["rank"] == 1
            ]
            top1.append(
                matches[0]["retrieved_text"]
                if matches else ""
            )

        if len(set(top1)) > 1:
            print(f"CHANGED: {q}")

if __name__ == "__main__":
    main()
