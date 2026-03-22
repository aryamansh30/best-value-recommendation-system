from __future__ import annotations

import math
from typing import Dict, List

from .types import Intent, NormalizedProduct, RankedProduct


def _minmax(values: List[float]) -> List[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


def _weights_for_intent(intent: Intent) -> Dict[str, float]:
    if intent == "cheapest":
        return {
            "price_efficiency": 0.55,
            "rating": 0.2,
            "relevance": 0.15,
            "reviews": 0.05,
            "discount": 0.05,
        }
    if intent == "premium":
        return {
            "price_efficiency": 0.1,
            "rating": 0.4,
            "relevance": 0.2,
            "reviews": 0.2,
            "discount": 0.1,
        }
    return {
        "price_efficiency": 0.3,
        "rating": 0.35,
        "relevance": 0.15,
        "reviews": 0.1,
        "discount": 0.1,
    }


def rank_products(products: List[NormalizedProduct], intent: Intent, top_k: int = 5) -> List[RankedProduct]:
    if not products:
        return []

    prices = [product.price for product in products]
    ratings = [product.rating for product in products]
    relevances = [product.relevance for product in products]
    review_signals = [math.log1p(max(product.reviews, 0)) for product in products]
    discounts = [product.discount for product in products]

    price_norm = _minmax(prices)
    rating_norm = _minmax(ratings)
    relevance_norm = _minmax(relevances)
    review_norm = _minmax(review_signals)
    discount_norm = _minmax(discounts)

    price_efficiency = [1.0 - value for value in price_norm]
    weights = _weights_for_intent(intent)

    ranked: List[RankedProduct] = []
    for idx, product in enumerate(products):
        components = {
            "price_efficiency": price_efficiency[idx],
            "rating": rating_norm[idx],
            "relevance": relevance_norm[idx],
            "reviews": review_norm[idx],
            "discount": discount_norm[idx],
        }

        weighted = {
            name: components[name] * weights[name] for name in components.keys()
        }
        final_score = sum(weighted.values())

        ranked.append(
            RankedProduct(
                product_id=product.product_id,
                title=product.title,
                category=product.category,
                price=product.price,
                rating=product.rating,
                reviews=product.reviews,
                description=product.description,
                source=product.source,
                relevance=product.relevance,
                score=round(final_score, 6),
                breakdown={
                    "final_score": round(final_score, 6),
                    "price_efficiency": round(weighted["price_efficiency"], 6),
                    "rating": round(weighted["rating"], 6),
                    "relevance": round(weighted["relevance"], 6),
                    "reviews": round(weighted["reviews"], 6),
                    "discount": round(weighted["discount"], 6),
                },
            )
        )

    ranked.sort(key=lambda item: (-item.score, item.price, -item.rating))
    return ranked[:top_k]
