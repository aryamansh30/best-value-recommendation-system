from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Optional

Intent = Literal["cheapest", "best_value", "premium", "general_search"]
ExplanationMode = Literal["deterministic", "rag_grounded"]


@dataclass
class ParsedQuery:
    query: str
    normalized_query: str
    category: Optional[str] = None
    budget: Optional[float] = None
    intent: Intent = "general_search"
    filters: Dict[str, str] = field(default_factory=dict)
    required_terms: List[str] = field(default_factory=list)
    used_genai: bool = False
    parser_notes: List[str] = field(default_factory=list)


@dataclass
class RawProduct:
    product_id: str
    title: str
    category: str
    price: Any
    rating: Any
    reviews: Any
    description: str
    source: str
    retrieval_score: float = 0.0
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    fused_score: float = 0.0
    discount: Optional[float] = None


@dataclass
class NormalizedProduct:
    product_id: str
    title: str
    category: str
    price: float
    rating: float
    reviews: int
    description: str
    source: str
    retrieval_score: float
    discount: float = 0.0
    relevance: float = 0.0
    lexical_score: float = 0.0
    semantic_score: float = 0.0
    fused_score: float = 0.0


@dataclass
class RankedProduct:
    product_id: str
    title: str
    category: str
    price: float
    rating: float
    reviews: int
    description: str
    source: str
    relevance: float
    score: float
    breakdown: Dict[str, float]
    llm_rerank_score: float = 0.0


@dataclass
class RecommendationResult:
    parsed_query: ParsedQuery
    candidates_considered: int
    shortlist: List[RankedProduct]
    best_value: Optional[RankedProduct]
    explanation: str
    provider_used: str
    source_trace: List[str]
    explanation_mode: ExplanationMode = "deterministic"
    citations: List[Dict[str, Any]] = field(default_factory=list)
    grounding: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
