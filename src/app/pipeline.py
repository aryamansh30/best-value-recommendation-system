from __future__ import annotations

from typing import List, Tuple

from .config import Settings
from .explainer import build_explanation
from .genai import GenAIClient
from .normalization import normalize_products
from .query_parser import CATEGORY_SYNONYMS, parse_query
from .ranking import rank_products
from .retrieval import ProductRetriever
from .types import NormalizedProduct, ParsedQuery, RecommendationResult


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

        explanation = build_explanation(
            parsed_query=parsed_query,
            best=best_value,
            runner_up=runner_up,
            candidates_considered=len(filtered_candidates),
            category_relaxed=filter_meta["category_relaxed"],
            genai_client=self.genai,
            allow_genai_rewrite=False,
        )

        return RecommendationResult(
            parsed_query=parsed_query,
            candidates_considered=len(filtered_candidates),
            shortlist=ranked,
            best_value=best_value,
            explanation=explanation,
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
