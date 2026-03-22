from __future__ import annotations

import json
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Tuple

from .config import Settings
from .types import ParsedQuery, RawProduct

FAKESTORE_CATEGORY_MAP: Dict[str, List[str]] = {
    "electronics": ["electronics"],
    "jewelry": ["jewelery"],
    "clothing": ["men's clothing", "women's clothing"],
    "headphones": ["electronics"],
    "keyboard": ["electronics"],
    "laptop": ["electronics"],
    "phone": ["electronics"],
}


class ProductRetriever:
    """Retrieves product candidates from required and optional sources."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ssl_context = self._build_ssl_context()
        self._fakestore_headers = {
            "Accept": "application/json",
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
            ),
        }

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
            source_trace.append("fakestore")

        include_rapidapi = use_rapidapi or self.settings.use_rapidapi
        if include_rapidapi:
            rapidapi_products = self._fetch_rapidapi(" ".join(query_terms).strip())
            raw_products.extend(rapidapi_products)
            if rapidapi_products:
                source_trace.append("rapidapi")

        scored = []
        for product in raw_products:
            retrieval_score = self._retrieval_score(product, query_terms, parsed_query.category)
            product.retrieval_score = retrieval_score
            scored.append(product)

        scored.sort(key=lambda item: item.retrieval_score, reverse=True)
        deduped = self._dedupe(scored)

        # If user query has meaningful terms but none match dataset content,
        # return empty candidates instead of ranking unrelated products.
        has_terms = any(term.strip() for term in query_terms)
        has_positive = any(product.retrieval_score > 0 for product in deduped)
        if has_terms and not has_positive:
            return [], source_trace or ["none"]

        if has_positive:
            deduped = [product for product in deduped if product.retrieval_score > 0]

        return deduped[:limit], source_trace or ["none"]

    def _fetch_fakestore(self, parsed_query: ParsedQuery, limit: int) -> List[RawProduct]:
        request_urls = self._build_fakestore_request_urls(parsed_query=parsed_query, limit=limit)
        merged: List[RawProduct] = []
        category_errors: List[str] = []

        for request_url in request_urls:
            request = urllib.request.Request(
                request_url,
                headers=self._fakestore_headers,
                method="GET",
            )
            try:
                with self._urlopen(request) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    merged.extend(self._adapt_fakestore(payload))
            except urllib.error.HTTPError as exc:
                if exc.code in {403, 404} and "/category/" in request_url:
                    category_errors.append(f"{request_url} -> HTTP {exc.code}")
                    continue
                raise RuntimeError(self._fakestore_http_error_message(exc)) from exc
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, ssl.SSLCertVerificationError):
                    raise RuntimeError(
                        "Failed to fetch live FakeStore API data due to SSL certificate verification. "
                        "Set SSL_CA_BUNDLE in .env to a valid CA file, or install certificates for your Python runtime."
                    ) from exc
                raise RuntimeError(
                    f"Failed to fetch live FakeStore API data ({exc.reason}). Snapshot fallback is disabled."
                ) from exc
            except json.JSONDecodeError:
                raise RuntimeError(
                    "Failed to fetch live FakeStore API data. Snapshot fallback is disabled."
                )

        if not merged:
            fallback_url = self._build_fakestore_global_url(limit=limit, intent=parsed_query.intent)
            request = urllib.request.Request(
                fallback_url,
                headers=self._fakestore_headers,
                method="GET",
            )
            try:
                with self._urlopen(request) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    merged.extend(self._adapt_fakestore(payload))
            except urllib.error.HTTPError as exc:
                if exc.code == 403:
                    base_url = self.settings.fakestore_url.split("?", 1)[0].rstrip("/")
                    base_request = urllib.request.Request(
                        base_url,
                        headers=self._fakestore_headers,
                        method="GET",
                    )
                    try:
                        with self._urlopen(base_request) as response:
                            payload = json.loads(response.read().decode("utf-8"))
                            merged.extend(self._adapt_fakestore(payload))
                    except urllib.error.HTTPError as base_exc:
                        raise RuntimeError(self._fakestore_http_error_message(base_exc)) from base_exc
                    except urllib.error.URLError as base_exc:
                        if isinstance(base_exc.reason, ssl.SSLCertVerificationError):
                            raise RuntimeError(
                                "Failed to fetch live FakeStore API data due to SSL certificate verification. "
                                "Set SSL_CA_BUNDLE in .env to a valid CA file, or install certificates for your Python runtime."
                            ) from base_exc
                        raise RuntimeError(
                            "Failed to fetch live FakeStore API data via global and base routes "
                            f"({base_exc.reason})."
                        ) from base_exc
                    except json.JSONDecodeError:
                        raise RuntimeError(
                            "Failed to fetch live FakeStore API data. Snapshot fallback is disabled."
                        )
                else:
                    raise RuntimeError(self._fakestore_http_error_message(exc)) from exc
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, ssl.SSLCertVerificationError):
                    raise RuntimeError(
                        "Failed to fetch live FakeStore API data due to SSL certificate verification. "
                        "Set SSL_CA_BUNDLE in .env to a valid CA file, or install certificates for your Python runtime."
                    ) from exc
                if category_errors:
                    raise RuntimeError(
                        "Failed to fetch live FakeStore API data via category routes and global route "
                        f"({exc.reason}). Category route errors: {', '.join(category_errors)}."
                    ) from exc
                raise RuntimeError(
                    f"Failed to fetch live FakeStore API data ({exc.reason}). Snapshot fallback is disabled."
                ) from exc
            except json.JSONDecodeError:
                raise RuntimeError(
                    "Failed to fetch live FakeStore API data. Snapshot fallback is disabled."
                )

        return self._dedupe(merged)

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

    def _build_fakestore_request_urls(self, parsed_query: ParsedQuery, limit: int) -> List[str]:
        """Use documented FakeStore GET routes for base/category retrieval."""
        base = self.settings.fakestore_url.split("?", 1)[0].rstrip("/")
        limit_value = max(1, limit)
        sort_value = self._sort_for_intent(parsed_query.intent)

        mapped_categories = FAKESTORE_CATEGORY_MAP.get((parsed_query.category or "").lower(), [])
        if not mapped_categories:
            params = urllib.parse.urlencode({"limit": str(limit_value), "sort": sort_value})
            return [f"{base}?{params}"]

        urls: List[str] = []
        for category in mapped_categories:
            encoded_category = urllib.parse.quote(category, safe="")
            params = urllib.parse.urlencode({"sort": sort_value})
            urls.append(f"{base}/category/{encoded_category}?{params}")
        return urls

    def _build_fakestore_global_url(self, limit: int, intent: str) -> str:
        base = self.settings.fakestore_url.split("?", 1)[0].rstrip("/")
        limit_value = max(1, limit)
        params = urllib.parse.urlencode({"limit": str(limit_value), "sort": self._sort_for_intent(intent)})
        return f"{base}?{params}"

    @staticmethod
    def _fakestore_http_error_message(exc: urllib.error.HTTPError) -> str:
        if exc.code == 403:
            return (
                "Failed to fetch live FakeStore API data (HTTP 403 Forbidden). "
                "Your network or upstream proxy likely blocks this request; "
                "try another network/VPN or allow outbound HTTPS to fakestoreapi.com."
            )
        return f"Failed to fetch live FakeStore API data (HTTP {exc.code}). Snapshot fallback is disabled."

    @staticmethod
    def _sort_for_intent(intent: str) -> str:
        if intent == "cheapest":
            return "asc"
        if intent == "premium":
            return "desc"
        return "desc"

    @staticmethod
    def _adapt_fakestore(payload: Iterable[Dict[str, object]]) -> List[RawProduct]:
        products = []
        for entry in payload:
            rating = entry.get("rating") if isinstance(entry, dict) else {}
            if not isinstance(rating, dict):
                rating = {}
            products.append(
                RawProduct(
                    product_id=str(entry.get("id", "")),
                    title=str(entry.get("title", "")).strip(),
                    category=str(entry.get("category", "")).strip().lower(),
                    price=entry.get("price", 0.0),
                    rating=rating.get("rate"),
                    reviews=rating.get("count"),
                    description=str(entry.get("description", "")).strip(),
                    source="fakestore",
                    discount=0.0,
                )
            )
        return products

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
    def _retrieval_score(product: RawProduct, query_terms: List[str], category: str | None) -> float:
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
