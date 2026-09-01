#!/usr/bin/env python3
"""
train_router_classifier_advanced_final.py

Train and validate the advanced Sentence-BERT + engineered-feature router on
the clean 160-query training dataset, then refit on ALL 160 training queries
and save the final production PKL.

IMPORTANT:
- The independent 60-query router test set must NOT be used here.
- The 60-query test set is reserved for final routing evaluation only.
"""

import argparse
import joblib
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "len",
    "word_count",
    "has_question_mark",
    "has_how_many",
    "has_compare",
    "has_navigate",
    "has_where",
    "has_what",
    "has_explain",
    "has_need",
]


def extract_features(query: str):
    q = query.lower()
    return {
        "len": len(query),
        "word_count": len(query.split()),
        "has_question_mark": float("?" in query),
        "has_how_many": float("how many" in q),
        "has_compare": float(
            any(
                w in q
                for w in [
                    "compare",
                    "which",
                    "highest",
                    "lowest",
                    "most",
                    "fewest",
                    "best",
                    "worst",
                    "difference",
                    "greater",
                    "less",
                    "average",
                    "total",
                    "combined",
                    "rank",
                    "order",
                    "fraction",
                    "percentage",
                ]
            )
        ),
        "has_navigate": float(
            any(w in q for w in ["navigate", "go to", "move to", "take", "send"])
        ),
        "has_where": float("where" in q),
        "has_what": float("what" in q),
        "has_explain": float(
            any(w in q for w in ["explain", "procedure", "describe", "tell me", "show"])
        ),
        "has_need": float(
            any(w in q for w in ["need", "should", "must", "required", "require"])
        ),
    }


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
        ),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-csv",
        default="router_training_clean_160.csv",
        help="Clean training CSV with columns: query,tier",
    )
    parser.add_argument(
        "--embedding-model",
        default="all-MiniLM-L6-v2",
        help="SentenceTransformer model used by the advanced ROUTER classifier",
    )
    parser.add_argument(
        "--output",
        default="router_classifier_advanced.pkl",
        help="Output PKL path",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)

    required = {"query", "tier"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.dropna(subset=["query", "tier"]).copy()
    df["query"] = df["query"].astype(str).str.strip()
    df["tier"] = df["tier"].astype(str).str.strip().str.lower()

    expected_labels = {"direct", "keyword", "rag", "llm"}
    unknown = set(df["tier"]) - expected_labels
    if unknown:
        raise ValueError(f"Unexpected tier labels: {sorted(unknown)}")

    queries = df["query"].values
    y = df["tier"].values

    print(f"Loaded {len(df)} training queries.")
    print("Class distribution:")
    print(df["tier"].value_counts().sort_index())

    # Engineered features
    features_df = pd.DataFrame([extract_features(q) for q in queries])
    features_df = features_df[FEATURE_NAMES]

    # Sentence-BERT embeddings for the ROUTER classifier
    embedder = SentenceTransformer(args.embedding_model)
    embeddings = embedder.encode(queries, show_progress_bar=True)

    X = np.hstack([embeddings, features_df.values])

    print(f"Embedding model: {args.embedding_model}")
    print("Embeddings shape:", embeddings.shape)
    print("Engineered features shape:", features_df.shape)
    print("Combined feature shape:", X.shape)

    # ----------------------------------------------------------
    # Development hold-out evaluation (training data only)
    # ----------------------------------------------------------
    X_train, X_dev, y_train, y_dev = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    dev_model = build_pipeline()
    dev_model.fit(X_train, y_train)
    y_pred = dev_model.predict(X_dev)

    print("\n=== Development hold-out report (20% of training set) ===")
    print(classification_report(y_dev, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_dev, y_pred, labels=["direct", "keyword", "rag", "llm"]))

    # ----------------------------------------------------------
    # 5-fold cross-validation on the 160-query training set
    # ----------------------------------------------------------
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_model = build_pipeline()
    cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy")

    print("\n=== 5-fold cross-validation (training set only) ===")
    print(f"Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print("Fold scores:", np.round(cv_scores, 4))

    # ----------------------------------------------------------
    # FINAL MODEL: refit on ALL 160 training queries
    # ----------------------------------------------------------
    final_pipeline = build_pipeline()
    final_pipeline.fit(X, y)

    package = {
        "embedder": embedder,
        "pipeline": final_pipeline,
        "features": FEATURE_NAMES,
        "embedding_model": args.embedding_model,
        "labels": list(final_pipeline.named_steps["clf"].classes_),
        "training_size": int(len(df)),
    }

    joblib.dump(package, args.output)

    print("\n=== Final production model ===")
    print(f"Refit on all {len(df)} training queries.")
    print(f"Saved: {args.output}")
    print("Do NOT tune using router_test_unseen_60.csv.")
    print("Apply the frozen runtime confidence threshold in router.py/config.py.")


if __name__ == "__main__":
    main()
