#!/usr/bin/env python3
"""
train_router_classifier_final.py

Train and validate the simple TF-IDF + Logistic Regression router on the
clean 160-query training dataset, then refit on ALL 160 training queries and
save the final production PKL.

IMPORTANT:
- The independent 60-query router test set must NOT be used here.
- The 60-query test set is reserved for final routing evaluation only.
"""

import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline


def build_pipeline():
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=5000,
                stop_words="english",
            ),
        ),
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
        "--output",
        default="router_classifier.pkl",
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

    X = df["query"].values
    y = df["tier"].values

    print(f"Loaded {len(df)} training queries.")
    print("Class distribution:")
    print(df["tier"].value_counts().sort_index())

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
    final_model = build_pipeline()
    final_model.fit(X, y)

    joblib.dump(final_model, args.output)

    print("\n=== Final production model ===")
    print(f"Refit on all {len(df)} training queries.")
    print(f"Saved: {args.output}")
    print("Do NOT tune using router_test_unseen_60.csv.")
    print("Apply the frozen runtime confidence threshold in router.py/config.py.")


if __name__ == "__main__":
    main()
