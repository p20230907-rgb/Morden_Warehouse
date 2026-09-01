# confidence_router.py
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum

class RoutingDecision(Enum):
    DIRECT = "direct"
    KEYWORD = "keyword"
    RAG = "rag"
    LLM = "llm"
    CLARIFY = "clarify"  # New: ask for clarification when confidence is low

@dataclass
class RoutingResult:
    tier: RoutingDecision
    confidence: float
    reasoning: str
    fallback_chain: List[str]

class ConfidenceRouter:
    """
    Confidence-aware router with fallback chain.
    Implements the "principled, confidence-aware routing mechanism"
    requested by Reviewer #1, point 3 and point 4b.
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.7,
                 fallback_threshold: float = 0.3):
        self.confidence_threshold = confidence_threshold
        self.fallback_threshold = fallback_threshold
        self.logger = logging.getLogger(__name__)
    
    def route(self, query: str, 
              shelf_locations: Dict,
              shelf_name_to_info: Dict,
              shelf_id_to_info: Dict,
              traditional_corpus: List[str]) -> RoutingResult:
        """
        Route query with confidence scoring and fallback chain.
        
        Implements:
        - Confidence estimation for each tier
        - Threshold-based gating (confidence < threshold → fallback)[reference:3]
        - Clarification when confidence is very low[reference:4]
        - Fallback chain: DIRECT → KEYWORD → RAG → LLM → CLARIFY
        """
        # Tier 1: Direct lookup confidence
        direct_confidence, direct_reasoning = self._direct_confidence(query, shelf_locations)
        if direct_confidence >= self.confidence_threshold:
            return RoutingResult(
                tier=RoutingDecision.DIRECT,
                confidence=direct_confidence,
                reasoning=direct_reasoning,
                fallback_chain=["DIRECT"]
            )
        
        # Tier 2: Keyword search confidence
        keyword_confidence, keyword_reasoning = self._keyword_confidence(query, traditional_corpus)
        if keyword_confidence >= self.confidence_threshold:
            return RoutingResult(
                tier=RoutingDecision.KEYWORD,
                confidence=keyword_confidence,
                reasoning=keyword_reasoning,
                fallback_chain=["DIRECT", "KEYWORD"]
            )
        
        # Tier 3: RAG confidence
        rag_confidence, rag_reasoning = self._rag_confidence(query)
        if rag_confidence >= self.confidence_threshold:
            return RoutingResult(
                tier=RoutingDecision.RAG,
                confidence=rag_confidence,
                reasoning=rag_reasoning,
                fallback_chain=["DIRECT", "KEYWORD", "RAG"]
            )
        
        # Tier 4: LLM reasoning confidence
        llm_confidence, llm_reasoning = self._llm_confidence(query)
        if llm_confidence >= self.confidence_threshold:
            return RoutingResult(
                tier=RoutingDecision.LLM,
                confidence=llm_confidence,
                reasoning=llm_reasoning,
                fallback_chain=["DIRECT", "KEYWORD", "RAG", "LLM"]
            )
        
        # If all confidences are low, ask for clarification
        return RoutingResult(
            tier=RoutingDecision.CLARIFY,
            confidence=max(direct_confidence, keyword_confidence, rag_confidence, llm_confidence),
            reasoning="All routing tiers have low confidence. Asking for clarification.",
            fallback_chain=["DIRECT", "KEYWORD", "RAG", "LLM", "CLARIFY"]
        )
    
    def _direct_confidence(self, query: str, shelf_locations: Dict) -> tuple:
        """Confidence score for direct lookup tier."""
        # Exact match → high confidence
        if query.lower() in [k.lower() for k in shelf_locations.keys()]:
            return 0.95, "Exact item match found"
        # Partial match → medium confidence
        for item in shelf_locations.keys():
            if query.lower() in item.lower() or item.lower() in query.lower():
                return 0.6, f"Partial match with '{item}'"
        return 0.1, "No direct match found"
    
    def _keyword_confidence(self, query: str, corpus: List[str]) -> tuple:
        """Confidence score for keyword search tier."""
        query_words = set(word.lower() for word in re.findall(r'\b\w+\b', query) if word.isalnum())
        if not query_words:
            return 0.0, "No keywords extracted"
        
        best_overlap = 0
        for text in corpus:
            text_words = set(word.lower() for word in re.findall(r'\b\w+\b', text) if word.isalnum())
            overlap = len(query_words.intersection(text_words))
            if overlap > best_overlap:
                best_overlap = overlap
        
        confidence = min(best_overlap / len(query_words), 1.0)
        reasoning = f"Keyword overlap: {best_overlap}/{len(query_words)} words matched"
        return confidence, reasoning
    
    def _rag_confidence(self, query: str) -> tuple:
        """Confidence score for RAG tier based on query complexity."""
        # More complex queries (more words, multiple conditions) → higher RAG confidence
        word_count = len(query.split())
        has_conditions = any(word in query.lower() for word in ["and", "or", "between", "than", "more", "less"])
        complexity_score = min(word_count / 10, 0.5) + (0.3 if has_conditions else 0)
        confidence = min(0.3 + complexity_score, 0.9)
        reasoning = f"Query complexity: {word_count} words, conditions: {has_conditions}"
        return confidence, reasoning
    
    def _llm_confidence(self, query: str) -> tuple:
        """Confidence score for LLM tier."""
        # LLM is the fallback for complex, ambiguous, or open-ended queries
        word_count = len(query.split())
        is_question = query.strip().endswith("?")
        confidence = min(0.4 + (word_count / 20) * 0.3 + (0.2 if is_question else 0), 0.85)
        reasoning = f"LLM as fallback for complex query: {word_count} words, question: {is_question}"
        return confidence, reasoning