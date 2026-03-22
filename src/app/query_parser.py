from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .types import ParsedQuery

STOPWORDS: Set[str] = {
    "a",
    "an",
    "and",
    "the",
    "for",
    "with",
    "under",
    "below",
    "best",
    "value",
    "cheap",
    "cheapest",
    "find",
    "show",
    "me",
    "to",
    "of",
    "in",
    "on",
    "by",
    "is",
    "are",
}

CATEGORY_SYNONYMS: Dict[str, List[str]] = {
    "headphones": ["headphone", "headphones", "earbuds", "earphones", "headset"],
    "keyboard": ["keyboard", "keyboards", "mechanical keyboard"],
    "protein bars": ["protein bar", "protein bars", "snack bar", "snack bars"],
    "laptop": ["laptop", "laptops"],
    "phone": ["phone", "smartphone", "mobile"],
    "electronics": ["electronics", "device", "gadget"],
    "clothing": ["shirt", "jacket", "clothing", "wear", "apparel"],
    "jewelry": ["jewelry", "jewelery", "ring", "necklace", "bracelet"],
}

INTENT_KEYWORDS = {
    "cheapest": ["cheapest", "cheap", "lowest", "budget", "low cost", "low-cost"],
    "best_value": ["best value", "best-value", "best", "value", "worth"],
    "premium": ["premium", "high end", "high-end", "luxury", "top tier", "top-tier"],
}

FILTER_KEYWORDS = {
    "wireless": ("type", "wireless"),
    "gaming": ("use_case", "gaming"),
    "beginner": ("experience", "beginner"),
    "portable": ("feature", "portable"),
}


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9$\s.-]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_budget(normalized_query: str) -> Optional[float]:
    patterns = [
        r"under\s*\$?\s*(\d+(?:\.\d+)?)",
        r"below\s*\$?\s*(\d+(?:\.\d+)?)",
        r"less than\s*\$?\s*(\d+(?:\.\d+)?)",
        r"up to\s*\$?\s*(\d+(?:\.\d+)?)",
        r"<=\s*\$?\s*(\d+(?:\.\d+)?)",
        r"\$\s*(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized_query)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def extract_intent(normalized_query: str) -> str:
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(keyword in normalized_query for keyword in keywords):
            return intent
    return "general_search"


def extract_filters(normalized_query: str) -> Dict[str, str]:
    filters: Dict[str, str] = {}
    for keyword, (field, value) in FILTER_KEYWORDS.items():
        if keyword in normalized_query:
            filters[field] = value
    return filters


def extract_category(normalized_query: str) -> Optional[str]:
    for canonical, synonyms in CATEGORY_SYNONYMS.items():
        for synonym in synonyms:
            if synonym in normalized_query:
                return canonical
    return None


def extract_required_terms(normalized_query: str) -> List[str]:
    terms = []
    for token in normalized_query.split():
        if token in STOPWORDS:
            continue
        if token.startswith("$"):
            continue
        if re.fullmatch(r"\d+(?:\.\d+)?", token):
            continue
        terms.append(token)
    return list(dict.fromkeys(terms))


def parse_query(query: str) -> ParsedQuery:
    normalized = normalize_text(query)
    parsed = ParsedQuery(
        query=query,
        normalized_query=normalized,
        category=extract_category(normalized),
        budget=extract_budget(normalized),
        intent=extract_intent(normalized),
        filters=extract_filters(normalized),
        required_terms=extract_required_terms(normalized),
    )
    if parsed.category is None:
        parsed.parser_notes.append("No direct category match from deterministic parser.")
    return parsed
