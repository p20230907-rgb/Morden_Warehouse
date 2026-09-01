#!/usr/bin/env python3
"""
test_router_classifier_advanced.py – Test the advanced classifier on custom queries.
"""

import joblib
import numpy as np
import re

# Load advanced model
data = joblib.load("router_classifier_advanced.pkl")
embedder = data["embedder"]
pipeline = data["pipeline"]

# Engineered features (must match the training script)
def extract_features(query):
    query_lower = query.lower()
    return {
        "len": len(query),
        "word_count": len(query.split()),
        "has_question_mark": int("?" in query),
        "has_how_many": int("how many" in query_lower),
        "has_compare": int(any(w in query_lower for w in ["compare", "which", "highest", "lowest", "most", "fewest"])),
        "has_navigate": int(any(w in query_lower for w in ["navigate", "go to", "move to", "take"])),
        "has_where": int("where" in query_lower),
        "has_what": int("what" in query_lower),
        "has_explain": int(any(w in query_lower for w in ["explain", "procedure", "describe"])),
        "has_need": int(any(w in query_lower for w in ["need", "should", "must", "required"])),
    }

test_queries = [
    "How many iPhones are in stock?",
    "What to do if there is a fire?",
    "Compare the prices of Dell laptops.",
    "forklift operators",
    "Safety Helmets",
    "Apple iPhone 14",
    "Explain the emergency evacuation procedure.",
]

print("Testing advanced classifier on new queries:\n")
for q in test_queries:
    features = extract_features(q)
    feat_vector = np.array([list(features.values())])
    emb = embedder.encode([q])
    X_input = np.hstack([emb, feat_vector])
    pred = pipeline.predict(X_input)[0]
    prob = pipeline.predict_proba(X_input)[0].max()
    print(f"{q:50} → {pred} (confidence: {prob:.2f})")