# retriever.py
import logging
from typing import List, Optional, Tuple

class HybridRetriever:
    def __init__(self, fusion_weights: dict):
        self.weights = fusion_weights

    def keyword_search(self, query: str, corpus: List[str]) -> Optional[str]:
        query_words = set(word.lower() for word in query.split() if word.isalnum())
        if not query_words:
            return None
        best_text = None
        best_score = 0
        for text in corpus:
            text_words = set(word.lower() for word in text.split() if word.isalnum())
            overlap = len(query_words.intersection(text_words))
            if overlap > best_score:
                best_score = overlap
                best_text = text
            if overlap == len(query_words):  # perfect match
                return best_text
        return best_text if best_score > 0 else None

    def semantic_search(self, query: str, vector_store, top_k: int = 3) -> List[str]:
        if vector_store is None:
            return []
        try:
            docs = vector_store.similarity_search(query, k=top_k)
            return [doc.page_content for doc in docs]
        except Exception as e:
            logging.error(f"Semantic search failed: {e}")
            return []

    def hybrid_search(self, query: str, corpus: List[str], vector_store) -> Tuple[Optional[str], float]:
        kw_text = self.keyword_search(query, corpus)
        sem_chunks = self.semantic_search(query, vector_store, top_k=1)
        sem_text = sem_chunks[0] if sem_chunks else None

        best_text = None
        best_score = 0.0
        if kw_text:
            score = self.weights['keyword'] * 1.0
            if score > best_score:
                best_score = score
                best_text = kw_text
        if sem_text:
            score = self.weights['semantic'] * 1.0
            if score > best_score:
                best_score = score
                best_text = sem_text
        return best_text, best_score