from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Tuple

from .config import Settings
from .explainer import build_explanation
from .genai import GenAIClient
from .normalization import normalize_products
from .query_parser import CATEGORY_SYNONYMS, parse_query
from .ranking import rank_products
from .retrieval import ProductRetriever
from .types import NormalizedProduct, ParsedQuery, RankedProduct, RecommendationResult


class RecommendationService:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings.from_env()
        self.retriever = ProductRetriever(self.settings)
        self.genai = GenAIClient(self.settings)

    def recommend(
        self,
        query: str,
        top_k: int = 5,
        use_rapidapi: bool = False,
    ) -> RecommendationResult:
        parsed_query = parse_query(query)
        query_terms = self._build_query_terms(parsed_query)
        raw_candidates, source_trace = self.retriever.retrieve(
            parsed_query=parsed_query,
            query_terms=query_terms,
            limit=max(20, top_k * 4),
            use_rapidapi=use_rapidapi,
        )
        retrieval_debug = getattr(self.retriever, "last_debug", {})
        normalized_candidates = normalize_products(raw_candidates)

        filtered_candidates, filter_meta = self._apply_constraints(
            normalized_candidates,
            parsed_query,
        )
        ranked = rank_products(filtered_candidates, parsed_query.intent, top_k=top_k)

        initial_filtered_count = len(filtered_candidates)
        genai_synonyms: List[str] = []
        genai_synonym_fallback_attempted = False
        genai_synonym_fallback_used = False

        # Deterministic first; trigger GenAI only when direct retrieval has no valid matches.
        if not ranked:
            genai_synonym_fallback_attempted = True
            genai_synonyms = self.genai.expand_synonyms(
                query=query,
                category=parsed_query.category,
                required_terms=parsed_query.required_terms,
                max_terms=5,
            )
            if genai_synonyms:
                parsed_query.used_genai = True
                parsed_query.parser_notes.append(
                    "GenAI synonym fallback triggered after deterministic no-match."
                )
                expanded_terms = list(dict.fromkeys(query_terms + genai_synonyms))
                raw_candidates_retry, source_trace_retry = self.retriever.retrieve(
                    parsed_query=parsed_query,
                    query_terms=expanded_terms,
                    limit=max(20, top_k * 4),
                    use_rapidapi=use_rapidapi,
                )
                normalized_retry = normalize_products(raw_candidates_retry)
                filtered_retry, filter_meta_retry = self._apply_constraints(
                    normalized_retry,
                    parsed_query,
                )
                ranked_retry = rank_products(filtered_retry, parsed_query.intent, top_k=top_k)

                if ranked_retry:
                    query_terms = expanded_terms
                    raw_candidates = raw_candidates_retry
                    normalized_candidates = normalized_retry
                    filtered_candidates = filtered_retry
                    filter_meta = filter_meta_retry
                    ranked = ranked_retry
                    source_trace = list(dict.fromkeys(source_trace + source_trace_retry))
                    genai_synonym_fallback_used = True

        best_value = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None
        rerank_meta = self._apply_optional_llm_rerank(
            query=query,
            parsed_query=parsed_query,
            ranked=ranked,
        )
        ranked = rerank_meta["ranked"]
        best_value = ranked[0] if ranked else None
        runner_up = ranked[1] if len(ranked) > 1 else None

        deterministic_explanation = build_explanation(
            parsed_query=parsed_query,
            best=best_value,
            runner_up=runner_up,
            candidates_considered=len(filtered_candidates),
            category_relaxed=filter_meta["category_relaxed"],
            genai_client=self.genai,
            allow_genai_rewrite=False,
        )
        if rerank_meta["used"]:
            deterministic_explanation += (
                " Final ordering includes optional LLM reranking on top deterministic candidates."
            )
        explanation = deterministic_explanation
        explanation_mode = "deterministic"
        citations: List[Dict[str, Any]] = []

        evidence_pack = self._build_evidence_pack(ranked, max_items=5)
        grounding: Dict[str, Any] = {
            "grounded_generation_used": False,
            "evidence_count": len(evidence_pack),
            "citation_count": 0,
            "fallback_reason": None,
        }
        rag_generation_attempted = False
        rag_used_evidence_ids: List[str] = []

        if ranked and evidence_pack:
            rag_generation_attempted = True
            try:
                grounded = self.genai.generate_grounded_explanation(
                    query=parsed_query.query,
                    intent=parsed_query.intent,
                    evidence=evidence_pack,
                    best_product={
                        "product_id": best_value.product_id if best_value else "",
                        "title": best_value.title if best_value else "",
                        "price": best_value.price if best_value else None,
                        "score": best_value.score if best_value else None,
                        "rerank_used": rerank_meta["used"],
                        "rerank_rationale": rerank_meta["rationale"],
                    },
                )
            except Exception:
                grounded = None
            if grounded and self._is_grounded_payload_valid(grounded, evidence_pack):
                explanation = grounded["explanation"]
                citations = grounded["citations"]
                explanation_mode = "rag_grounded"
                parsed_query.used_genai = True
                rag_used_evidence_ids = grounded.get("used_evidence_ids", [])
                grounding = {
                    "grounded_generation_used": True,
                    "evidence_count": len(evidence_pack),
                    "citation_count": len(citations),
                    "fallback_reason": None,
                }
            else:
                grounding["fallback_reason"] = "invalid_or_unavailable_grounded_generation"
        elif not ranked:
            grounding["fallback_reason"] = "no_ranked_candidates"
        else:
            grounding["fallback_reason"] = "no_evidence_available"

        return RecommendationResult(
            parsed_query=parsed_query,
            candidates_considered=len(filtered_candidates),
            shortlist=ranked,
            best_value=best_value,
            explanation=explanation,
            explanation_mode=explanation_mode,
            citations=citations,
            grounding=grounding,
            provider_used=self.settings.active_provider() if parsed_query.used_genai else "none",
            source_trace=source_trace,
            debug={
                "query_terms": query_terms,
                "raw_candidates": len(raw_candidates),
                "normalized_candidates": len(normalized_candidates),
                "initial_filtered_candidates": initial_filtered_count,
                "genai_synonym_fallback_attempted": genai_synonym_fallback_attempted,
                "genai_synonym_fallback_used": genai_synonym_fallback_used,
                "genai_synonyms": genai_synonyms,
                "rag_generation_attempted": rag_generation_attempted,
                "rag_generation_used": explanation_mode == "rag_grounded",
                "rag_used_evidence_ids": rag_used_evidence_ids,
                "rag_evidence_pack": evidence_pack,
                "rag_fallback_reason": grounding["fallback_reason"],
                "rerank_attempted": rerank_meta["attempted"],
                "rerank_used": rerank_meta["used"],
                "rerank_fallback_reason": rerank_meta["fallback_reason"],
                "rerank_weight": rerank_meta["weight"],
                "rerank_top_n": rerank_meta["top_n"],
                "rerank_rationale": rerank_meta["rationale"],
                "rerank_citations": rerank_meta["citations"],
                "retrieval_lexical_weight": retrieval_debug.get("lexical_weight"),
                "retrieval_semantic_weight": retrieval_debug.get("semantic_weight"),
                "retrieval_semantic_fallback_reason": retrieval_debug.get("semantic_fallback_reason"),
                "retrieval_breakdown": retrieval_debug.get("retrieval_breakdown", []),
                **filter_meta,
            },
        )

    @staticmethod
    def _build_query_terms(parsed_query: ParsedQuery) -> List[str]:
        terms = list(parsed_query.required_terms)
        if parsed_query.category:
            terms.extend(parsed_query.category.split())
            terms.extend(CATEGORY_SYNONYMS.get(parsed_query.category, []))
        # Keep list unique while preserving order.
        return list(dict.fromkeys([term.lower() for term in terms if term.strip()]))

    def _apply_constraints(
        self,
        products: List[NormalizedProduct],
        parsed_query: ParsedQuery,
    ) -> Tuple[List[NormalizedProduct], dict]:
        budget_filtered = products
        if parsed_query.budget is not None:
            budget_filtered = [product for product in products if product.price <= parsed_query.budget]

        strict_category_filtered = budget_filtered
        category_relaxed = False
        no_category_match = False

        if parsed_query.category:
            strict_category_filtered = [
                product
                for product in budget_filtered
                if self._matches_category(product, parsed_query.category)
            ]

            # For explicit categories, we do not relax to unrelated products.
            if not strict_category_filtered:
                no_category_match = True

        return strict_category_filtered, {
            "budget_filtered": len(budget_filtered),
            "category_filtered": len(strict_category_filtered),
            "category_relaxed": category_relaxed,
            "no_category_match": no_category_match,
        }

    @staticmethod
    def _matches_category(product: NormalizedProduct, category: str) -> bool:
        text = f"{product.title.lower()} {product.category.lower()} {product.description.lower()}"

        if category in text:
            return True

        synonyms = CATEGORY_SYNONYMS.get(category, [])
        return any(term in text for term in synonyms)

    @staticmethod
    def _build_evidence_pack(
        ranked: List[RankedProduct],
        max_items: int = 5,
    ) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for idx, product in enumerate(ranked[: max(0, max_items)], start=1):
            snippet = product.description.strip()
            if len(snippet) > 180:
                snippet = snippet[:177].rstrip() + "..."
            evidence.append(
                {
                    "evidence_id": f"E{idx}",
                    "product_id": product.product_id,
                    "title": product.title,
                    "price": round(product.price, 2),
                    "rating": round(product.rating, 2),
                    "reviews": product.reviews,
                    "source": product.source,
                    "description_snippet": snippet,
                }
            )
        return evidence

    @staticmethod
    def _is_grounded_payload_valid(payload: Dict[str, Any], evidence: List[Dict[str, Any]]) -> bool:
        explanation = payload.get("explanation")
        citations = payload.get("citations")
        used_ids = payload.get("used_evidence_ids", [])
        if not isinstance(explanation, str) or not explanation.strip():
            return False
        if not isinstance(citations, list) or not citations:
            return False
        if not isinstance(used_ids, list):
            return False

        valid_ids = {
            str(item.get("evidence_id", "")).strip()
            for item in evidence
            if isinstance(item, dict) and str(item.get("evidence_id", "")).strip()
        }
        if not valid_ids:
            return False

        citation_ids: List[str] = []
        for citation in citations:
            if not isinstance(citation, dict):
                return False
            evidence_id = str(citation.get("evidence_id", "")).strip()
            product_id = str(citation.get("product_id", "")).strip()
            field = str(citation.get("field", "")).strip()
            quote_or_value = str(citation.get("quote_or_value", "")).strip()
            if not evidence_id or evidence_id not in valid_ids:
                return False
            if not product_id or not field or not quote_or_value:
                return False
            citation_ids.append(evidence_id)

        for used_id in used_ids:
            if str(used_id).strip() not in valid_ids:
                return False

        cleaned = explanation.strip()
        if not all(f"[{evidence_id}]" in cleaned for evidence_id in citation_ids):
            return False
        return True

    def _apply_optional_llm_rerank(
        self,
        query: str,
        parsed_query: ParsedQuery,
        ranked: List[RankedProduct],
    ) -> Dict[str, Any]:
        top_n = max(1, min(self.settings.llm_rerank_top_n, len(ranked) if ranked else 1))
        weight = min(max(self.settings.llm_rerank_weight, 0.0), 1.0)
        meta = {
            "ranked": ranked,
            "attempted": False,
            "used": False,
            "fallback_reason": None,
            "weight": round(weight, 6),
            "top_n": top_n,
            "rationale": "",
            "citations": [],
        }
        if not ranked:
            meta["fallback_reason"] = "no_ranked_candidates"
            return meta
        if not self.settings.enable_llm_rerank:
            meta["fallback_reason"] = "rerank_disabled"
            return meta

        subset = ranked[:top_n]
        payload_candidates = [
            {
                "product_id": item.product_id,
                "title": item.title,
                "price": round(item.price, 2),
                "rating": round(item.rating, 2),
                "reviews": item.reviews,
                "relevance": round(item.relevance, 6),
                "deterministic_score": round(item.score, 6),
                "description": item.description,
            }
            for item in subset
        ]
        meta["attempted"] = True

        try:
            reranked = self.genai.rerank_candidates(
                query=query,
                intent=parsed_query.intent,
                candidates=payload_candidates,
            )
        except Exception:
            meta["fallback_reason"] = "rerank_exception"
            return meta

        if not reranked:
            meta["fallback_reason"] = "rerank_invalid_or_unavailable"
            return meta

        ordered_ids = reranked.get("ordered_product_ids", [])
        if not isinstance(ordered_ids, list) or len(ordered_ids) != len(subset):
            meta["fallback_reason"] = "rerank_invalid_order"
            return meta
        subset_map = {item.product_id: item for item in subset}
        if set(ordered_ids) != set(subset_map.keys()):
            meta["fallback_reason"] = "rerank_invalid_order"
            return meta

        den = max(len(ordered_ids) - 1, 1)
        adjusted: List[RankedProduct] = []
        for rank_idx, product_id in enumerate(ordered_ids):
            original = subset_map[product_id]
            llm_rank_score = 1.0 if len(ordered_ids) == 1 else (1.0 - (rank_idx / den))
            combined = ((1.0 - weight) * original.score) + (weight * llm_rank_score)
            next_breakdown = dict(original.breakdown)
            next_breakdown["deterministic_base"] = round(original.score, 6)
            next_breakdown["llm_rerank"] = round(llm_rank_score * weight, 6)
            next_breakdown["final_score"] = round(combined, 6)
            adjusted.append(
                replace(
                    original,
                    score=round(combined, 6),
                    llm_rerank_score=round(llm_rank_score, 6),
                    breakdown=next_breakdown,
                )
            )

        adjusted.sort(key=lambda item: (-item.score, item.price, -item.rating))
        meta["ranked"] = adjusted + ranked[top_n:]
        meta["used"] = True
        meta["fallback_reason"] = None
        meta["rationale"] = str(reranked.get("rationale", "")).strip()
        raw_citations = reranked.get("citations", [])
        if isinstance(raw_citations, list):
            meta["citations"] = [entry for entry in raw_citations if isinstance(entry, str)]
        parsed_query.used_genai = True
        return meta
