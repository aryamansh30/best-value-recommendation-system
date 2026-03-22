from __future__ import annotations

from typing import Optional

from .genai import GenAIClient
from .types import ParsedQuery, RankedProduct


def build_explanation(
    parsed_query: ParsedQuery,
    best: Optional[RankedProduct],
    runner_up: Optional[RankedProduct],
    candidates_considered: int,
    category_relaxed: bool,
    genai_client: Optional[GenAIClient] = None,
    allow_genai_rewrite: bool = False,
) -> str:
    if best is None:
        category_text = parsed_query.category or "requested criteria"
        budget_text = (
            f" within budget ${parsed_query.budget:.2f}" if parsed_query.budget is not None else ""
        )
        return (
            f"No product found in the current dataset for '{category_text}'{budget_text}. "
            "Deterministic filtering and scoring were not run because no valid candidates matched."
        )

    budget_clause = (
        f" within your budget of ${parsed_query.budget:.2f}" if parsed_query.budget is not None else ""
    )
    fallback_clause = (
        " Category matching was relaxed due to sparse direct matches in the dataset."
        if category_relaxed
        else ""
    )

    explanation = (
        f"Top recommendation: '{best.title}' at ${best.price:.2f}{budget_clause}. "
        f"It ranked highest on deterministic best-value scoring (score={best.score:.3f}) using "
        f"price efficiency, rating, relevance, review confidence, and discount signals across "
        f"{candidates_considered} candidate products."
    )

    if runner_up is not None:
        explanation += (
            f" It beat '{runner_up.title}' mainly via stronger weighted components: "
            f"rating={best.breakdown['rating']:.3f} vs {runner_up.breakdown['rating']:.3f}, "
            f"price_efficiency={best.breakdown['price_efficiency']:.3f} vs "
            f"{runner_up.breakdown['price_efficiency']:.3f}."
        )

    explanation += fallback_clause

    if allow_genai_rewrite and genai_client and genai_client.enabled:
        rewritten = genai_client.rewrite_explanation(
            explanation,
            {
                "query": parsed_query.query,
                "intent": parsed_query.intent,
                "best_product": best.title,
                "best_score": best.score,
            },
        )
        return rewritten

    return explanation
