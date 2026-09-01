# router.py
"""
Router module for the warehouse assistant.
Supports:
  - "rule_based" / "confidence" : original heuristic router
  - "classifier" : TF‑IDF + LogisticRegression (router_classifier.pkl)
  - "classifier_advanced" : Sentence‑BERT + engineered features (router_classifier_advanced.pkl)
"""

import os
import re
import logging
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict

# For classifier modes
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# For advanced classifier
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

# Safety keywords used in rule-based mode
SAFETY_KEYWORDS = [
    "safety", "ppe", "fire", "hazardous", "loto", "manual",
    "procedure", "report", "training", "emergency", "exit",
    "first aid", "evacuation", "incident", "injury",
    "spill", "containment", "chemical", "hazard"
]

# ----------------------------------------------------------------------
# Helper: engineered features for advanced classifier
# ----------------------------------------------------------------------
def extract_features(query: str) -> Dict[str, float]:
    """Extract 10 engineered features from a query."""
    q = query.lower()
    return {
        "len": len(query),
        "word_count": len(query.split()),
        "has_question_mark": float("?" in query),
        "has_how_many": float("how many" in q),
        "has_compare": float(any(w in q for w in ["compare", "which", "highest", "lowest", "most", "fewest", "best", "worst"])),
        "has_navigate": float(any(w in q for w in ["navigate", "go to", "move to", "take", "send", "go"])),
        "has_where": float("where" in q),
        "has_what": float("what" in q),
        "has_explain": float(any(w in q for w in ["explain", "procedure", "describe", "tell me", "show"])),
        "has_need": float(any(w in q for w in ["need", "should", "must", "required", "require"])),
    }


class Router:
    """
    Main router class. Supports multiple routing modes.
    """

    def __init__(self, mode: str = "rule_based",
                 training_data_path: Optional[str] = None,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.7,
                 fallback_threshold: float = 0.3):
        # Map old "confidence" to "rule_based" for backward compatibility
        if mode == "confidence":
            mode = "rule_based"
        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.fallback_threshold = fallback_threshold

        self.classifier = None
        self.vectorizer = None
        self.embedder = None
        self.pipeline = None
        self.feature_names = None
        self.labels = None

        if mode == "classifier":
            self._load_simple_classifier(training_data_path)
        elif mode == "classifier_advanced":
            self._load_advanced_classifier(model_path)
        else:
            logging.info("Router mode: rule_based (heuristic)")

    # ------------------------------------------------------------------
    # Model loading methods
    # ------------------------------------------------------------------
    def _load_simple_classifier(self, training_data_path: Optional[str] = None):
        """Load TF‑IDF classifier from disk or train from CSV."""
        if os.path.exists("router_classifier.pkl"):
            try:
                pipeline = joblib.load("router_classifier.pkl")
                self.vectorizer = pipeline.named_steps["tfidf"]
                self.classifier = pipeline.named_steps["clf"]
                self.labels = self.classifier.classes_
                logging.info("Loaded TF‑IDF classifier from router_classifier.pkl")
                return
            except Exception as e:
                logging.warning(f"Could not load router_classifier.pkl: {e}")

        if training_data_path is None or not os.path.exists(training_data_path):
            logging.warning("No training data provided. Falling back to rule_based.")
            self.mode = "rule_based"
            return

        df = pd.read_csv(training_data_path)
        X = df["query"].values
        y = df["tier"].values
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, stop_words="english")
        self.classifier = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        self.classifier.fit(self.vectorizer.fit_transform(X), y)
        self.labels = self.classifier.classes_
        logging.info("Trained TF‑IDF classifier from CSV.")

    def _load_advanced_classifier(self, model_path: Optional[str] = None):
        """Load advanced Sentence‑BERT classifier from disk."""
        if model_path is None:
            model_path = "router_classifier_advanced.pkl"
        if not os.path.exists(model_path):
            logging.warning(f"Advanced model not found at {model_path}. Falling back to rule_based.")
            self.mode = "rule_based"
            return
        try:
            data = joblib.load(model_path)
            self.embedder = data["embedder"]
            self.pipeline = data["pipeline"]
            self.feature_names = data.get("features", [])
            self.classifier = self.pipeline.named_steps.get("clf")
            if self.classifier is not None:
                self.labels = self.classifier.classes_
            else:
                self.labels = data.get("labels", ["direct", "keyword", "rag", "llm"])
            logging.info("Loaded advanced Sentence‑BERT classifier from %s", model_path)
        except Exception as e:
            logging.warning(f"Failed to load advanced model: {e}. Falling back to rule_based.")
            self.mode = "rule_based"

    # ------------------------------------------------------------------
    # Prediction methods
    # ------------------------------------------------------------------
    def predict(self, query: str, shelf_locations: Dict, safety_keywords: List[str]) -> Tuple[str, float]:
        """
        Route the query to one of the tiers.
        Returns: (tier, confidence)
        """
        if self.mode == "classifier" and self.classifier is not None:
            return self._predict_classifier(query)
        elif self.mode == "classifier_advanced" and self.pipeline is not None:
            return self._predict_advanced(query)
        else:
            return self._predict_rule_based(query, shelf_locations, safety_keywords)

    # ------------------------------------------------------------------
    # Rule-based prediction (original logic)
    # ------------------------------------------------------------------
    def _predict_rule_based(self, query: str, shelf_locations: Dict, safety_keywords: List[str]) -> Tuple[str, float]:
        q = query.strip().lower()

        # 1. Direct lookup
        if q in [k.lower() for k in shelf_locations.keys()]:
            return "direct", 0.95
        for item in shelf_locations.keys():
            if q in item.lower() or item.lower() in q:
                return "direct", 0.60
        if q.startswith("shelf ") and any(q in name.lower() for name in shelf_locations.keys()):
            return "direct", 0.90
        if len(q) == 1 and q.upper() in [loc.get("shelf_id") for loc in shelf_locations.values()]:
            return "direct", 0.90

        # 2. Safety → force RAG (skip keyword)
        if any(kw in q for kw in safety_keywords):
            return "rag", 0.85

        # 3. Keyword search (heuristic)
        if len(q.split()) > 2 and not q.endswith("?"):
            return "keyword", 0.75

        # 4. LLM fallback
        return "llm", 0.70

    # ------------------------------------------------------------------
    # Simple TF‑IDF classifier prediction
    # ------------------------------------------------------------------
    def _predict_classifier(self, query: str) -> Tuple[str, float]:
        if self.classifier is None:
            return "rag", 0.5
        X_vec = self.vectorizer.transform([query])
        probs = self.classifier.predict_proba(X_vec)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        tier = self.labels[pred_idx]
        if confidence < self.confidence_threshold:
            return "rag", confidence
        return tier, confidence

    # ------------------------------------------------------------------
    # Advanced Sentence‑BERT classifier prediction
    # ------------------------------------------------------------------
    def _predict_advanced(self, query: str) -> Tuple[str, float]:
        if self.pipeline is None or self.embedder is None:
            return "rag", 0.5
        feat_dict = extract_features(query)
        feat_vector = np.array([list(feat_dict.values())])
        emb = self.embedder.encode([query])
        X_input = np.hstack([emb, feat_vector])
        probs = self.pipeline.predict_proba(X_input)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        tier = self.labels[pred_idx]
        if confidence < self.confidence_threshold:
            return "rag", confidence
        return tier, confidence

    # ------------------------------------------------------------------
    # Static helpers for reproducibility (Reviewer #1, point 4a)
    # ------------------------------------------------------------------
    @staticmethod
    def is_safety_or_sop(query: str) -> bool:
        return any(kw in query.lower() for kw in SAFETY_KEYWORDS)

    @staticmethod
    def is_keyword_answerable(query: str, corpus: List[str], threshold: float = 0.5) -> bool:
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        if not query_words:
            return False
        for text in corpus:
            text_words = set(re.findall(r'\b\w+\b', text.lower()))
            overlap = len(query_words.intersection(text_words))
            if overlap / len(query_words) >= threshold:
                return True
        return False