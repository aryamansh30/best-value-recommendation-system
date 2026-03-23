from __future__ import annotations

import json
import re
import ssl
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from .config import Settings
from .query_parser import CATEGORY_SYNONYMS


class GenAIError(RuntimeError):
    """Raised when a GenAI call fails."""


class GenAIClient:
    """Selective GenAI helper for targeted fallback tasks."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ssl_context = self._build_ssl_context()

    @property
    def enabled(self) -> bool:
        return True

    def expand_synonyms(
        self,
        query: str,
        category: str | None,
        required_terms: List[str],
        max_terms: int = 5,
    ) -> List[str]:
        prompt = (
            "Generate closely related shopping search terms for fallback retrieval. "
            "Return strict JSON with key 'synonyms' as an array of 2-5 short terms. "
            "Rules: keep terms narrowly related to the requested product type, "
            "do not broaden to unrelated categories, do not include generic words like product/item."
        )
        payload = {
            "query": query,
            "category": category,
            "required_terms": required_terms,
            "max_terms": max_terms,
        }
        try:
            content = self._chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                force_json=True,
            )
        except GenAIError:
            return []

        if not content:
            return []
        parsed = self._extract_json(content)
        if not parsed:
            return []

        raw_synonyms = parsed.get("synonyms")
        if not isinstance(raw_synonyms, list):
            return []

        blocked = {"product", "products", "item", "items", "thing", "stuff"}
        broad = {
            "electronics",
            "clothing",
            "jewelry",
            "jewelery",
            "fashion",
            "accessories",
            "device",
            "gadget",
        }
        allowed_hint_terms = {term.lower().strip() for term in required_terms if term.strip()}
        if category:
            allowed_hint_terms.add(category.lower().strip())
            allowed_hint_terms.update(CATEGORY_SYNONYMS.get(category.lower().strip(), []))

        cleaned: List[str] = []
        for entry in raw_synonyms:
            if not isinstance(entry, str):
                continue
            term = re.sub(r"[^a-z0-9\s-]", " ", entry.lower()).strip()
            term = re.sub(r"\s+", " ", term)
            if not term:
                continue
            words = [word for word in term.split() if word]
            if len(words) > 3:
                continue
            if any(word in blocked or word in broad for word in words):
                continue
            # Guardrail: for explicit categories, keep terms lexically close to known intent tokens.
            if allowed_hint_terms and not any(word in allowed_hint_terms for word in words):
                # Allow short singular/plural variations.
                singularized = {word[:-1] if word.endswith("s") else word for word in words}
                if not singularized.intersection(allowed_hint_terms):
                    continue
            cleaned.append(term)

        # Deduplicate, preserve order, and cap.
        result: List[str] = []
        seen = set()
        for term in cleaned:
            if term in seen:
                continue
            seen.add(term)
            result.append(term)
            if len(result) >= max_terms:
                break
        return result

    def extract_query_fields(self, query: str) -> Optional[Dict[str, Any]]:
        prompt = (
            "Extract structured shopping fields from the user query. "
            "Return JSON only with keys: category, budget, intent, filters. "
            "Intent must be one of: cheapest, best_value, premium, general_search. "
            "If unknown, use null or {}."
        )
        user_message = f"Query: {query}"
        content = self._chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_message},
            ],
            force_json=True,
        )
        if not content:
            raise GenAIError("GenAI parser call failed: no response from configured provider.")

        payload = self._extract_json(content)
        if not payload:
            raise GenAIError("GenAI parser call failed: model response did not contain valid JSON.")

        result: Dict[str, Any] = {
            "category": payload.get("category"),
            "budget": payload.get("budget"),
            "intent": payload.get("intent"),
            "filters": payload.get("filters", {}),
        }
        return result

    def rewrite_explanation(self, explanation: str, context: Dict[str, Any]) -> str:
        prompt = (
            "Rewrite the recommendation explanation for clarity. "
            "Keep facts unchanged and concise (2-3 sentences)."
        )
        content = self._chat(
            [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": json.dumps({"explanation": explanation, "context": context}),
                },
            ],
            force_json=False,
        )
        if not content:
            raise GenAIError("GenAI explanation rewrite failed: no response from configured provider.")
        cleaned = self._extract_explanation_text(content)
        return cleaned.strip() if cleaned.strip() else content.strip()

    def generate_grounded_explanation(
        self,
        query: str,
        intent: str,
        evidence: List[Dict[str, Any]],
        best_product: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        if not evidence:
            return None

        prompt = (
            "You are generating a shopping recommendation explanation using retrieval-augmented evidence only. "
            "Rules: Use only the supplied evidence objects, do not add outside facts, and cite evidence IDs inline "
            "as [E#]. Every factual statement must be backed by at least one citation. "
            "Return strict JSON with keys: explanation, citations, used_evidence_ids, limitations. "
            "Citations must be an array of objects with keys: evidence_id, product_id, field, quote_or_value."
        )
        payload = {
            "query": query,
            "intent": intent,
            "best_product": best_product or {},
            "evidence": evidence,
        }
        try:
            content = self._chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                force_json=True,
            )
        except GenAIError:
            return self._deterministic_grounded_fallback(query=query, intent=intent, evidence=evidence)

        if not content:
            return self._deterministic_grounded_fallback(query=query, intent=intent, evidence=evidence)

        parsed = self._extract_json(content)
        if not parsed:
            return self._deterministic_grounded_fallback(query=query, intent=intent, evidence=evidence)

        validated = self._validate_grounded_payload(parsed, evidence)
        if validated:
            return validated
        return self._deterministic_grounded_fallback(query=query, intent=intent, evidence=evidence)

    def rerank_candidates(
        self,
        query: str,
        intent: str,
        candidates: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None
        candidate_ids = [str(item.get("product_id", "")).strip() for item in candidates if isinstance(item, dict)]
        if not candidate_ids or any(not product_id for product_id in candidate_ids):
            return None

        prompt = (
            "Re-rank shopping candidates using ONLY the provided candidate data and requested intent. "
            "Do not add outside facts. Return strict JSON with keys: ordered_product_ids, rationale, citations. "
            "ordered_product_ids must include every provided product_id exactly once in ranked order. "
            "rationale should be concise text. citations should be optional and may be an array of short strings."
        )
        payload = {
            "query": query,
            "intent": intent,
            "candidates": candidates,
        }
        try:
            content = self._chat(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(payload)},
                ],
                force_json=True,
            )
        except GenAIError:
            return self._deterministic_rerank_fallback(candidates)
        if not content:
            return self._deterministic_rerank_fallback(candidates)
        parsed = self._extract_json(content)
        if not parsed:
            return self._deterministic_rerank_fallback(candidates)
        validated = self._validate_rerank_payload(parsed, candidate_ids)
        if validated:
            return validated
        return self._deterministic_rerank_fallback(candidates)

    def _chat(self, messages: List[Dict[str, str]], force_json: bool = False) -> Optional[str]:
        provider = self.settings.llm_provider
        if provider == "openai":
            return self._chat_openai(messages, force_json=force_json)
        return self._chat_ollama(messages, force_json=force_json)

    def _chat_ollama(self, messages: List[Dict[str, str]], force_json: bool = False) -> Optional[str]:
        url = self.settings.ollama_base_url.rstrip("/") + "/api/chat"
        payload = {
            "model": self.settings.ollama_model,
            "messages": messages,
            "stream": False,
        }
        if force_json:
            payload["format"] = "json"
        return self._http_json_request(url, payload, headers={})

    def _chat_openai(self, messages: List[Dict[str, str]], force_json: bool = False) -> Optional[str]:
        if not self.settings.openai_api_key:
            raise GenAIError("OpenAI provider selected but OPENAI_API_KEY is not set.")
        url = self.settings.openai_base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": self.settings.openai_model,
            "messages": messages,
            "temperature": 0.1,
        }
        if force_json:
            payload["response_format"] = {"type": "json_object"}
        return self._http_json_request(
            url,
            payload,
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            openai_mode=True,
        )

    def _http_json_request(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        openai_mode: bool = False,
    ) -> Optional[str]:
        base_headers = {"Content-Type": "application/json"}
        base_headers.update(headers)
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=base_headers, method="POST")
        try:
            with self._urlopen(request) as response:
                raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if openai_mode:
                    return parsed["choices"][0]["message"]["content"]
                return parsed.get("message", {}).get("content")
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, ssl.SSLCertVerificationError):
                raise GenAIError(
                    "GenAI HTTPS request failed SSL verification. "
                    "Set SSL_CA_BUNDLE in .env to a valid CA file, or install certificates for your Python runtime."
                ) from exc
            return None
        except (urllib.error.HTTPError, json.JSONDecodeError, KeyError, IndexError):
            return None

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
    def _extract_json(text: str) -> Optional[Dict[str, Any]]:
        text = text.strip()
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    @staticmethod
    def _extract_explanation_text(text: str) -> str:
        stripped = text.strip()
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return stripped

        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, dict):
            for key in ("recommendation", "explanation", "text", "message", "content"):
                value = parsed.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return stripped

    @staticmethod
    def _validate_grounded_payload(
        payload: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        explanation = payload.get("explanation")
        raw_citations = payload.get("citations")
        raw_used_ids = payload.get("used_evidence_ids")
        limitations = payload.get("limitations", "")

        if not isinstance(explanation, str) or not explanation.strip():
            return None
        if not isinstance(raw_citations, list) or not raw_citations:
            return None
        if raw_used_ids is None:
            raw_used_ids = []
        if not isinstance(raw_used_ids, list):
            return None

        valid_evidence_ids = {
            str(item.get("evidence_id", "")).strip()
            for item in evidence
            if isinstance(item, dict) and str(item.get("evidence_id", "")).strip()
        }
        if not valid_evidence_ids:
            return None

        citations: List[Dict[str, str]] = []
        citation_ids: List[str] = []
        for entry in raw_citations:
            if not isinstance(entry, dict):
                return None
            evidence_id = str(entry.get("evidence_id", "")).strip()
            product_id = str(entry.get("product_id", "")).strip()
            field = str(entry.get("field", "")).strip()
            quote_or_value = str(entry.get("quote_or_value", "")).strip()

            if not evidence_id or evidence_id not in valid_evidence_ids:
                return None
            if not product_id or not field or not quote_or_value:
                return None

            citation_ids.append(evidence_id)
            citations.append(
                {
                    "evidence_id": evidence_id,
                    "product_id": product_id,
                    "field": field,
                    "quote_or_value": quote_or_value,
                }
            )

        used_evidence_ids: List[str] = []
        seen = set()
        for item in raw_used_ids:
            evidence_id = str(item).strip()
            if not evidence_id:
                continue
            if evidence_id not in valid_evidence_ids:
                return None
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            used_evidence_ids.append(evidence_id)

        if not used_evidence_ids:
            for evidence_id in citation_ids:
                if evidence_id not in seen:
                    seen.add(evidence_id)
                    used_evidence_ids.append(evidence_id)

        cleaned_explanation = explanation.strip()
        if not any(f"[{evidence_id}]" in cleaned_explanation for evidence_id in used_evidence_ids):
            return None
        if not all(f"[{evidence_id}]" in cleaned_explanation for evidence_id in citation_ids):
            return None

        limitations_text = limitations.strip() if isinstance(limitations, str) else ""
        return {
            "explanation": cleaned_explanation,
            "citations": citations,
            "used_evidence_ids": used_evidence_ids,
            "limitations": limitations_text,
        }

    @staticmethod
    def _validate_rerank_payload(
        payload: Dict[str, Any],
        candidate_ids: List[str],
    ) -> Optional[Dict[str, Any]]:
        raw_ordered = payload.get("ordered_product_ids")
        if not isinstance(raw_ordered, list):
            return None
        ordered_ids: List[str] = []
        for entry in raw_ordered:
            product_id = str(entry).strip()
            if not product_id:
                return None
            ordered_ids.append(product_id)
        if len(ordered_ids) != len(candidate_ids):
            return None
        if len(set(ordered_ids)) != len(ordered_ids):
            return None
        if set(ordered_ids) != set(candidate_ids):
            return None

        rationale = payload.get("rationale", "")
        if not isinstance(rationale, str):
            rationale = ""

        raw_citations = payload.get("citations", [])
        citations: List[str] = []
        if isinstance(raw_citations, list):
            for entry in raw_citations:
                if isinstance(entry, str) and entry.strip():
                    citations.append(entry.strip())

        return {
            "ordered_product_ids": ordered_ids,
            "rationale": rationale.strip(),
            "citations": citations,
        }

    @staticmethod
    def _deterministic_grounded_fallback(
        query: str,
        intent: str,
        evidence: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not evidence:
            return None
        first = evidence[0]
        e1 = str(first.get("evidence_id", "E1")).strip() or "E1"
        p1 = str(first.get("product_id", "")).strip()
        t1 = str(first.get("title", "")).strip()
        price1 = first.get("price")
        rating1 = first.get("rating")
        reviews1 = first.get("reviews")

        citations: List[Dict[str, str]] = []
        explanation_bits: List[str] = []

        if p1 and price1 is not None:
            citations.append(
                {
                    "evidence_id": e1,
                    "product_id": p1,
                    "field": "price",
                    "quote_or_value": str(price1),
                }
            )
            explanation_bits.append(f"{t1} is priced at ${price1} [{e1}]")

        if p1 and rating1 is not None:
            citations.append(
                {
                    "evidence_id": e1,
                    "product_id": p1,
                    "field": "rating",
                    "quote_or_value": str(rating1),
                }
            )
            explanation_bits.append(f"with rating {rating1} [{e1}]")

        if p1 and reviews1 is not None:
            citations.append(
                {
                    "evidence_id": e1,
                    "product_id": p1,
                    "field": "reviews",
                    "quote_or_value": str(reviews1),
                }
            )
            explanation_bits.append(f"and {reviews1} reviews [{e1}]")

        if len(evidence) > 1:
            second = evidence[1]
            e2 = str(second.get("evidence_id", "E2")).strip() or "E2"
            p2 = str(second.get("product_id", "")).strip()
            t2 = str(second.get("title", "")).strip()
            price2 = second.get("price")
            if p2 and price2 is not None:
                citations.append(
                    {
                        "evidence_id": e2,
                        "product_id": p2,
                        "field": "price",
                        "quote_or_value": str(price2),
                    }
                )
                explanation_bits.append(f"A close alternative is {t2} at ${price2} [{e2}]")

        if not citations:
            return None

        explanation = (
            f"For '{query}' ({intent}), "
            + "; ".join(explanation_bits)
            + ". Recommendation is grounded only in retrieved catalog evidence."
        )
        used_evidence_ids: List[str] = []
        seen = set()
        for entry in citations:
            evidence_id = entry["evidence_id"]
            if evidence_id in seen:
                continue
            seen.add(evidence_id)
            used_evidence_ids.append(evidence_id)

        return {
            "explanation": explanation.strip(),
            "citations": citations,
            "used_evidence_ids": used_evidence_ids,
            "limitations": "Generated from retrieved catalog evidence only.",
        }

    @staticmethod
    def _deterministic_rerank_fallback(candidates: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not candidates:
            return None

        def _score(entry: Dict[str, Any]) -> tuple[float, float, float]:
            det = float(entry.get("deterministic_score") or 0.0)
            rating = float(entry.get("rating") or 0.0)
            price = float(entry.get("price") or 0.0)
            return (det, rating, -price)

        ordered = sorted(
            [entry for entry in candidates if isinstance(entry, dict) and str(entry.get("product_id", "")).strip()],
            key=_score,
            reverse=True,
        )
        ordered_ids = [str(entry["product_id"]).strip() for entry in ordered]
        if not ordered_ids:
            return None
        return {
            "ordered_product_ids": ordered_ids,
            "rationale": "Deterministic fallback rerank based on deterministic score, rating, and price.",
            "citations": ["local_fallback"],
        }
