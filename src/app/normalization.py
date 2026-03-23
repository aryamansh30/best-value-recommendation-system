from __future__ import annotations

import re
from typing import Iterable, List

from .types import NormalizedProduct, RawProduct


def _to_float(value, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    text = re.sub(r"[^0-9.]+", "", text)
    if not text:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def _to_int(value, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = re.sub(r"[^0-9]", "", str(value))
    return int(text) if text else default


def _clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"[\s\n\t]+", " ", text)
    text = re.sub(r"[!]{2,}", "!", text)
    return text


def normalize_products(raw_products: Iterable[RawProduct]) -> List[NormalizedProduct]:
    normalized = []
    for product in raw_products:
        price = _to_float(product.price, default=0.0)
        rating = _to_float(product.rating, default=3.5)
        reviews = _to_int(product.reviews, default=0)
        discount = _to_float(product.discount, default=0.0)

        # Bound common fields to realistic ranges.
        rating = min(max(rating, 0.0), 5.0)
        if price < 0:
            price = 0.0
        if discount < 0:
            discount = 0.0

        normalized.append(
            NormalizedProduct(
                product_id=product.product_id,
                title=_clean_text(product.title),
                category=_clean_text(product.category).lower(),
                price=price,
                rating=rating,
                reviews=reviews,
                description=_clean_text(product.description),
                source=product.source,
                retrieval_score=product.retrieval_score,
                discount=discount,
                relevance=product.retrieval_score,
                lexical_score=product.lexical_score,
                semantic_score=product.semantic_score,
                fused_score=product.fused_score or product.retrieval_score,
            )
        )
    return normalized
