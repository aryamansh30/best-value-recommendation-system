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
