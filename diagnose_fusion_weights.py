#!/usr/bin/env python3
import argparse, csv
import numpy as np
from pathlib import Path
from config import CONFIG
from data_loader import load_data

WEIGHT_SETS = [
    {"semantic":0.6,"keyword":0.3,"metadata":0.1},
    {"semantic":0.7,"keyword":0.2,"metadata":0.1},
    {"semantic":0.5,"keyword":0.4,"metadata":0.1},
    {"semantic":0.4,"keyword":0.3,"metadata":0.3},
]

DEFAULT_QUERIES = [
    "orange reflective vest",
    "helmet use in loading zones",
    "What should a visitor wear before entering operational warehouse areas?",
    "What are the documented requirements for a safe chemical-storage cabinet?",
    "What inspection schedule applies to firefighting equipment?",
    "What PPE is required when employees handle sharp objects?"
]

def short(text, n=160):
    if not text: return ""
    s = " ".join(str(text).split())
    return s if len(s) <= n else s[:n] + "..."

def keyword_search(query, corpus):
    qwords = set(w.lower() for w in query.split() if w.isalnum())
    if not qwords: return None
    best_text, best_score = None, 0
    for text in corpus:
        twords = set(w.lower() for w in text.split() if w.isalnum())
        overlap = len(qwords.intersection(twords))
        if overlap > best_score:
            best_score, best_text = overlap, text
        if overlap == len(qwords):
            return best_text
    return best_text if best_score > 0 else None

def semantic_search_direct(query, vector_store, top_k=1):
    ef = vector_store.embedding_function
    if hasattr(ef, "embed_query"):
        qvec = ef.embed_query(query)
    elif callable(ef):
        qvec = ef(query)
    else:
        raise RuntimeError("embedding_function cannot create a query embedding")
    qvec = np.array([qvec], dtype="float32")
    D, I = vector_store.index.search(qvec, top_k)
    out = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0: continue
        doc_id = vector_store.index_to_docstore_id[idx]
        doc = vector_store.docstore.search(doc_id)
        out.append((getattr(doc, "page_content", str(doc)), float(dist)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", action="append")
    ap.add_argument("--output", default="fusion_weight_diagnostic_v2.csv")
    args = ap.parse_args()
    queries = args.query or DEFAULT_QUERIES

    base = Path(__file__).resolve().parent
    data = load_data(str(base), CONFIG)
    corpus = data["traditional_search_corpus"]
    vs = data["vector_store"]
    if vs is None: raise RuntimeError("Vector store is None")

    rows, changed_count = [], 0

    for i, query in enumerate(queries, 1):
        print(f"\nQUERY {i}: {query}")
        selected_all = []

        for w in WEIGHT_SETS:
            kw = keyword_search(query, corpus)
            sems = semantic_search_direct(query, vs, top_k=1)
            sem = sems[0][0] if sems else None
            dist = sems[0][1] if sems else None

            selected, score, source = None, 0.0, "none"
            if kw:
                s = w["keyword"] * 1.0
                if s > score:
                    selected, score, source = kw, s, "keyword"
            if sem:
                s = w["semantic"] * 1.0
                if s > score:
                    selected, score, source = sem, s, "semantic"

            selected_all.append(selected or "")
            print(f"  S/K/M={w['semantic']}/{w['keyword']}/{w['metadata']}")
            print(f"    keyword : {short(kw)}")
            print(f"    semantic: {short(sem)}")
            print(f"    distance: {dist}")
            print(f"    selected: {source}, score={score:.3f}")

            rows.append({
                "query": query,
                "semantic_weight": w["semantic"],
                "keyword_weight": w["keyword"],
                "metadata_weight": w["metadata"],
                "keyword_result": kw or "",
                "semantic_result": sem or "",
                "faiss_distance": dist,
                "selected_source": source,
                "selected_score": score,
                "selected_result": selected or ""
            })

        changed = len(set(selected_all)) > 1
        changed_count += int(changed)
        print(f"  Selected passage changed across weights? {changed}")

    with open(args.output, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nChanged selections: {changed_count}/{len(queries)}")
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
