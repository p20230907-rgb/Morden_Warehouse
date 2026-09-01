#!/usr/bin/env python3
"""
test_router_classifier.py – Test the saved classifier on custom queries.
"""

import joblib

# Load the trained model
model = joblib.load("router_classifier.pkl")

test_queries = [
    "How many iPhones are in stock?",
    "What to do if there is a fire?",
    "Compare the prices of Dell laptops.",
    "forklift operators",
    "Safety Helmets",
    "Apple iPhone 14",
    "Explain the emergency evacuation procedure.",
]

print("Testing classifier on new queries:\n")
for q in test_queries:
    pred = model.predict([q])[0]
    prob = model.predict_proba([q])[0].max()
    print(f"{q:50} → {pred} (confidence: {prob:.2f})")