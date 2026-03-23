from __future__ import annotations

import csv
import json
import ssl
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

from .config import Settings
from .embeddings import EmbeddingClient, cosine_similarity, normalize_weight_pair
from .types import ParsedQuery, RawProduct

FAKESTORE_CATEGORY_MAP: Dict[str, List[str]] = {
    "electronics": ["electronics"],
    "jewelry": ["jewelery"],
    "clothing": ["men's clothing", "women's clothing"],
    "protein bars": ["grocery"],
    "grocery": ["grocery"],
    "beauty": ["beauty"],
    "headphones": ["electronics"],
    "keyboard": ["electronics"],
    "laptop": ["electronics"],
    "phone": ["electronics"],
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ProductRetriever:
    """Retrieves product candidates from required and optional sources."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ssl_context = self._build_ssl_context()
        self._embedding_client = EmbeddingClient(settings=settings, ssl_context=self._ssl_context)
        self._catalog_rows: Optional[List[Dict[str, str]]] = None
        self.last_debug: Dict[str, Any] = {}

    def retrieve(
        self,
        parsed_query: ParsedQuery,
        query_terms: List[str],
        limit: int = 50,
        use_rapidapi: bool = False,
    ) -> Tuple[List[RawProduct], List[str]]:
        source_trace: List[str] = []

        raw_products = self._fetch_fakestore(parsed_query=parsed_query, limit=limit)
        if raw_products:
            source_trace.append("catalog_csv")

        include_rapidapi = use_rapidapi or self.settings.use_rapidapi
        if include_rapidapi:
            rapidapi_products = self._fetch_rapidapi(" ".join(query_terms).strip())
            raw_products.extend(rapidapi_products)
            if rapidapi_products:
                source_trace.append("rapidapi")

        deduped = self._dedupe(raw_products)
        semantic_scores, semantic_meta = self._semantic_scores(
            parsed_query=parsed_query,
            query_terms=query_terms,
            products=deduped,
        )
        lexical_weight, semantic_weight = normalize_weight_pair(
            lexical_weight=self.settings.lexical_weight,
            semantic_weight=self.settings.semantic_weight,
        )
        force_lexical_only = bool(semantic_meta.get("force_lexical_only"))
        if force_lexical_only:
            lexical_weight = 1.0
            semantic_weight = 0.0

        for product in deduped:
            lexical_score = self._lexical_retrieval_score(product, query_terms, parsed_query.category)
            semantic_score = semantic_scores.get(product.product_id, 0.0)
            fused_score = (
                lexical_score
                if force_lexical_only
                else (lexical_weight * lexical_score + semantic_weight * semantic_score)
            )
            product.lexical_score = round(lexical_score, 6)
            product.semantic_score = round(semantic_score, 6)
            product.fused_score = round(fused_score, 6)
            product.retrieval_score = round(fused_score, 6)

        deduped.sort(key=lambda item: item.retrieval_score, reverse=True)

        # If user query has meaningful terms but none match dataset content,
        # return empty candidates instead of ranking unrelated products.
        has_terms = any(term.strip() for term in query_terms)
        has_positive = any(product.retrieval_score > 0 for product in deduped)
        runtime_descriptor = getattr(self._embedding_client, "runtime_descriptor", None)
        if callable(runtime_descriptor):
            embedding_provider, embedding_model = runtime_descriptor()
        else:
            embedding_provider = getattr(self._embedding_client, "provider", "unknown")
            embedding_model = getattr(self._embedding_client, "model", "unknown")
        self.last_debug = {
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
            "semantic_fallback_reason": semantic_meta.get("reason"),
            "force_lexical_only": force_lexical_only,
            "lexical_weight": round(lexical_weight, 6),
            "semantic_weight": round(semantic_weight, 6),
            "semantic_candidate_count": semantic_meta.get("semantic_candidate_count", 0),
            "retrieval_breakdown": [
                {
                    "product_id": product.product_id,
                    "title": product.title,
                    "lexical_score": product.lexical_score,
                    "semantic_score": product.semantic_score,
                    "fused_score": product.fused_score,
                }
                for product in deduped[: min(len(deduped), limit)]
            ],
        }

        if has_terms and not has_positive:
            return [], source_trace or ["none"]

        if has_positive:
            deduped = [product for product in deduped if product.retrieval_score > 0]

        return deduped[:limit], source_trace or ["none"]

    def _fetch_fakestore(self, parsed_query: ParsedQuery, limit: int) -> List[RawProduct]:
        rows = self._load_catalog_rows()
        if not rows:
            return []

        mapped_categories = self._mapped_categories(parsed_query.category)
        filtered_rows = rows
        if mapped_categories:
            filtered_rows = [row for row in rows if row.get("category", "").strip().lower() in mapped_categories]
            if not filtered_rows:
                filtered_rows = rows

        products = [self._adapt_catalog_row(row) for row in filtered_rows if self._is_in_stock(row.get("in_stock"))]
        descending = self._sort_for_intent(parsed_query.intent) == "desc"
        products.sort(key=lambda item: self._safe_float(item.price), reverse=descending)
        return self._dedupe(products)[: max(1, limit)]

    def _fetch_rapidapi(self, query: str) -> List[RawProduct]:
        if not self.settings.rapidapi_key or not query:
            return []

        params = urllib.parse.urlencode({"query": query, "country": self.settings.rapidapi_region})
        url = f"{self.settings.rapidapi_url}?{params}"
        headers = {
            "X-RapidAPI-Key": self.settings.rapidapi_key,
            "X-RapidAPI-Host": self.settings.rapidapi_host,
        }
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with self._urlopen(request) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return self._adapt_rapidapi(payload)
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return []

    def _build_ssl_context(self) -> Optional[ssl.SSLContext]:
        if self.settings.ssl_ca_bundle:
            return ssl.create_default_context(cafile=self.settings.ssl_ca_bundle)

        try:
            import certifi

            return ssl.create_default_context(cafile=certifi.where())
        except Exception:
            return None

    def _urlopen(self, request: urllib.request.Request):
        if self._ssl_context is None:
            return urllib.request.urlopen(request, timeout=self.settings.request_timeout_sec)
        return urllib.request.urlopen(
            request,
            timeout=self.settings.request_timeout_sec,
            context=self._ssl_context,
        )

    @staticmethod
    def _sort_for_intent(intent: str) -> str:
        if intent == "cheapest":
            return "asc"
        if intent == "premium":
            return "desc"
        return "desc"

    def _load_catalog_rows(self) -> List[Dict[str, str]]:
        if self._catalog_rows is not None:
            return self._catalog_rows

        csv_path = self._catalog_path()
        if not csv_path.exists():
            raise RuntimeError(
                f"Catalog CSV not found at '{csv_path}'. Set CATALOG_CSV_PATH in .env to a valid file."
            )

        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None:
                    raise RuntimeError(f"Catalog CSV '{csv_path}' has no header row.")
                required = {"id", "title", "price", "category", "rating_rate", "rating_count", "description"}
                missing = sorted(required.difference(set(reader.fieldnames)))
                if missing:
                    raise RuntimeError(
                        f"Catalog CSV '{csv_path}' is missing required columns: {', '.join(missing)}"
                    )

                rows: List[Dict[str, str]] = []
                for row in reader:
                    if not row:
                        continue
                    if not str(row.get("id", "")).strip():
                        continue
                    rows.append(row)
                self._catalog_rows = rows
                return rows
        except OSError as exc:
            raise RuntimeError(f"Failed to read catalog CSV '{csv_path}': {exc}") from exc

    def _catalog_path(self) -> Path:
        configured = Path(self.settings.catalog_csv_path).expanduser()
        if configured.is_absolute():
            return configured
        return (PROJECT_ROOT / configured).resolve()

    @staticmethod
    def _mapped_categories(category: Optional[str]) -> List[str]:
        normalized = (category or "").strip().lower()
        if not normalized:
            return []
        mapped = FAKESTORE_CATEGORY_MAP.get(normalized)
        if mapped:
            return [value.strip().lower() for value in mapped]
        return [normalized]

    @staticmethod
    def _safe_float(value: object, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _is_in_stock(value: object) -> bool:
        if value is None:
            return True
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _adapt_catalog_row(entry: Dict[str, str]) -> RawProduct:
        return RawProduct(
            product_id=str(entry.get("id", "")).strip(),
            title=str(entry.get("title", "")).strip(),
            category=str(entry.get("category", "")).strip().lower(),
            price=entry.get("price", 0.0),
            rating=entry.get("rating_rate", 0.0),
            reviews=entry.get("rating_count", 0),
            description=str(entry.get("description", "")).strip(),
            source="catalog_csv",
            discount=entry.get("discount_percent", 0.0),
        )

    @staticmethod
    def _adapt_rapidapi(payload: Dict[str, object]) -> List[RawProduct]:
        products = []
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            return products
        entries = data.get("products")
        if not isinstance(entries, list):
            return products

        for idx, entry in enumerate(entries):
            if not isinstance(entry, dict):
                continue
            price_text = entry.get("product_price") or entry.get("product_original_price") or 0
            rating_text = entry.get("product_star_rating")
            reviews_text = entry.get("product_num_ratings")
            discount_text = entry.get("product_offer")

            products.append(
                RawProduct(
                    product_id=str(entry.get("asin") or f"rapidapi-{idx}"),
                    title=str(entry.get("product_title", "")).strip(),
                    category=str(entry.get("product_category", "")).strip().lower(),
                    price=price_text,
                    rating=rating_text,
                    reviews=reviews_text,
                    description=str(entry.get("product_description") or "").strip(),
                    source="rapidapi",
                    discount=discount_text,
                )
            )
        return products

    @staticmethod
    def _lexical_retrieval_score(product: RawProduct, query_terms: List[str], category: str | None) -> float:
        if not query_terms:
            return 0.2

        haystack = " ".join(
            [product.title.lower(), product.category.lower(), product.description.lower()]
        )
        title_text = product.title.lower()

        overlap = sum(1 for term in query_terms if term in haystack)
        overlap_score = overlap / max(len(query_terms), 1)

        title_boost = sum(1 for term in query_terms if term in title_text) / max(len(query_terms), 1)

        category_boost = 0.0
        if category and category.lower() in haystack:
            category_boost = 1.0

        return 0.5 * overlap_score + 0.3 * title_boost + 0.2 * category_boost

    def _semantic_scores(
        self,
        parsed_query: ParsedQuery,
        query_terms: List[str],
        products: List[RawProduct],
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        if not products:
            return {}, {
                "force_lexical_only": True,
                "reason": "no_products",
                "semantic_candidate_count": 0,
            }

        if self.settings.semantic_weight <= 0:
            return {}, {
                "force_lexical_only": True,
                "reason": "semantic_weight_disabled",
                "semantic_candidate_count": 0,
            }
        if not self._embedding_client.available():
            return {}, {
                "force_lexical_only": True,
                "reason": "embedding_provider_unavailable",
                "semantic_candidate_count": 0,
            }

        query_terms_text = " ".join(term for term in query_terms if term.strip()).strip()
        query_text = " | ".join(part for part in [parsed_query.query.strip(), query_terms_text] if part).strip()
        if not query_text:
            return {}, {
                "force_lexical_only": True,
                "reason": "empty_query_text",
                "semantic_candidate_count": 0,
            }

        query_vector = self._embedding_client.embed_text(query_text, cache_id=f"query:{query_text}")
        if not query_vector:
            return {}, {
                "force_lexical_only": True,
                "reason": "query_embedding_unavailable",
                "semantic_candidate_count": 0,
            }

        scores: Dict[str, float] = {}
        for product in products:
            product_text = self._embedding_text(product)
            product_vector = self._embedding_client.embed_text(
                product_text,
                cache_id=f"product:{product.product_id}",
            )
            if not product_vector:
                continue
            if len(product_vector) != len(query_vector):
                continue
            raw_similarity = cosine_similarity(query_vector, product_vector)
            similarity_01 = (raw_similarity + 1.0) / 2.0
            scores[product.product_id] = max(0.0, min(1.0, similarity_01))

        if not scores:
            return {}, {
                "force_lexical_only": True,
                "reason": "no_valid_product_embeddings",
                "semantic_candidate_count": 0,
            }
        return scores, {
            "force_lexical_only": False,
            "reason": None,
            "semantic_candidate_count": len(scores),
        }

    @staticmethod
    def _embedding_text(product: RawProduct) -> str:
        return " | ".join(
            part.strip()
            for part in [product.title, product.category, product.description]
            if part and part.strip()
        )

    @staticmethod
    def _dedupe(products: List[RawProduct]) -> List[RawProduct]:
        seen = set()
        deduped = []
        for product in products:
            key = (product.title.lower().strip(), str(product.price).strip())
            if key in seen:
                continue
            seen.add(key)
            deduped.append(product)
        return deduped
