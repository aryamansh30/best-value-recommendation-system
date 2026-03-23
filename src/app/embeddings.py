from __future__ import annotations

import hashlib
import json
import math
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

from .config import Settings


def cosine_similarity(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for l_val, r_val in zip(left, right):
        dot += l_val * r_val
        left_norm += l_val * l_val
        right_norm += r_val * r_val
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    score = dot / (math.sqrt(left_norm) * math.sqrt(right_norm))
    if not math.isfinite(score):
        return 0.0
    return max(-1.0, min(1.0, score))


def normalize_weight_pair(lexical_weight: float, semantic_weight: float) -> tuple[float, float]:
    lexical = max(0.0, float(lexical_weight))
    semantic = max(0.0, float(semantic_weight))
    total = lexical + semantic
    if total <= 0.0:
        return 1.0, 0.0
    return lexical / total, semantic / total


class EmbeddingCache:
    def __init__(self, path: str):
        self.path = Path(path)
        self._loaded = False
        self._values: Dict[str, List[float]] = {}

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not self.path.exists():
            self._values = {}
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            self._values = {}
            return
        if not isinstance(payload, dict):
            self._values = {}
            return

        loaded: Dict[str, List[float]] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, list):
                continue
            vector: List[float] = []
            valid = True
            for entry in value:
                if not isinstance(entry, (float, int)):
                    valid = False
                    break
                number = float(entry)
                if not math.isfinite(number):
                    valid = False
                    break
                vector.append(number)
            if valid and vector:
                loaded[key] = vector
        self._values = loaded

    def get(self, key: str) -> Optional[List[float]]:
        self._load()
        value = self._values.get(key)
        return list(value) if value else None

    def set(self, key: str, value: List[float]) -> None:
        self._load()
        self._values[key] = list(value)
        self._persist()

    def _persist(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._values), encoding="utf-8")
        except OSError:
            # Cache persistence failure should never break retrieval behavior.
            return


class EmbeddingClient:
    def __init__(self, settings: Settings, ssl_context: Optional[ssl.SSLContext]):
        self.settings = settings
        self.ssl_context = ssl_context
        self.cache = EmbeddingCache(settings.embedding_cache_path)
        self.provider = (settings.embedding_provider or "openai").strip().lower()
        self.model = self._resolve_model()
        self._local_dim = 256

    def _resolve_model(self) -> str:
        if self.provider == "ollama":
            return self.settings.ollama_embedding_model
        if self.provider == "openai":
            return self.settings.openai_embedding_model
        return self.settings.local_embedding_model

    def available(self) -> bool:
        if self.provider == "local":
            return True
        if self.provider == "openai":
            return bool(self.settings.openai_api_key.strip()) or self.settings.embedding_local_fallback
        if self.provider == "ollama":
            return bool(self.settings.ollama_base_url.strip()) or self.settings.embedding_local_fallback
        return self.settings.embedding_local_fallback

    def embed_text(self, text: str, cache_id: str) -> Optional[List[float]]:
        normalized_text = text.strip()
        if not normalized_text:
            return None
        mode, model = self._runtime_mode_and_model()
        if mode == "none":
            return None
        cache_key = self._cache_key(cache_id=cache_id, text=normalized_text, mode=mode, model=model)
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        if mode == "ollama":
            vector = self._embed_ollama(normalized_text)
        elif mode == "openai":
            vector = self._embed_openai(normalized_text)
        else:
            vector = self._embed_local(normalized_text)
        if (not self._valid_vector(vector)) and mode in {"openai", "ollama"} and self.settings.embedding_local_fallback:
            vector = self._embed_local(normalized_text)
        if not self._valid_vector(vector):
            return None

        self.cache.set(cache_key, vector)
        return vector

    def _cache_key(self, cache_id: str, text: str, mode: str, model: str) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{mode}:{model}:{cache_id}:{digest}"

    def _runtime_mode_and_model(self) -> tuple[str, str]:
        if self.provider == "local":
            return "local", self.settings.local_embedding_model
        if self.provider == "openai":
            if self.settings.openai_api_key.strip():
                return "openai", self.settings.openai_embedding_model
            if self.settings.embedding_local_fallback:
                return "local", self.settings.local_embedding_model
            return "none", ""
        if self.provider == "ollama":
            if self.settings.ollama_base_url.strip():
                return "ollama", self.settings.ollama_embedding_model
            if self.settings.embedding_local_fallback:
                return "local", self.settings.local_embedding_model
            return "none", ""
        if self.settings.embedding_local_fallback:
            return "local", self.settings.local_embedding_model
        return "none", ""

    def runtime_descriptor(self) -> tuple[str, str]:
        mode, model = self._runtime_mode_and_model()
        if mode == "none":
            return self.provider, self.model
        return mode, model

    def _embed_openai(self, text: str) -> Optional[List[float]]:
        if not self.settings.openai_api_key:
            return None
        url = self.settings.openai_base_url.rstrip("/") + "/embeddings"
        payload = {
            "model": self.settings.openai_embedding_model,
            "input": text,
        }
        parsed = self._post_json(
            url=url,
            payload=payload,
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
        )
        if not isinstance(parsed, dict):
            return None
        data = parsed.get("data")
        if not isinstance(data, list) or not data:
            return None
        first = data[0]
        if not isinstance(first, dict):
            return None
        embedding = first.get("embedding")
        if not isinstance(embedding, list):
            return None
        return self._coerce_vector(embedding)

    def _embed_ollama(self, text: str) -> Optional[List[float]]:
        base = self.settings.ollama_base_url.rstrip("/")

        # Current Ollama route for single prompt embeddings.
        parsed = self._post_json(
            url=f"{base}/api/embeddings",
            payload={"model": self.settings.ollama_embedding_model, "prompt": text},
            headers={},
        )
        vector = self._extract_ollama_embedding(parsed)
        if vector:
            return vector

        # Compatibility route for newer API shape.
        parsed = self._post_json(
            url=f"{base}/api/embed",
            payload={"model": self.settings.ollama_embedding_model, "input": [text]},
            headers={},
        )
        if not isinstance(parsed, dict):
            return None
        embeddings = parsed.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            return None
        first = embeddings[0]
        if not isinstance(first, list):
            return None
        return self._coerce_vector(first)

    def _embed_local(self, text: str) -> Optional[List[float]]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        if not tokens:
            return None
        vector = [0.0] * self._local_dim
        for token in tokens:
            base_digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(base_digest[:2], "big") % self._local_dim
            sign = 1.0 if (base_digest[2] % 2 == 0) else -1.0
            # Weight longer tokens slightly higher to reduce dominance of short common terms.
            weight = 1.0 + (min(len(token), 12) / 24.0)
            vector[index] += sign * weight

        norm = math.sqrt(sum(value * value for value in vector))
        if norm <= 0.0:
            return None
        return [value / norm for value in vector]

    def _extract_ollama_embedding(self, parsed: object) -> Optional[List[float]]:
        if not isinstance(parsed, dict):
            return None
        embedding = parsed.get("embedding")
        if not isinstance(embedding, list):
            return None
        return self._coerce_vector(embedding)

    def _post_json(self, url: str, payload: Dict[str, object], headers: Dict[str, str]) -> object:
        body = json.dumps(payload).encode("utf-8")
        request_headers = {"Content-Type": "application/json"}
        request_headers.update(headers)
        request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")
        try:
            with self._urlopen(request) as response:
                raw = response.read().decode("utf-8")
        except (urllib.error.HTTPError, urllib.error.URLError):
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def _urlopen(self, request: urllib.request.Request):
        if self.ssl_context is None:
            return urllib.request.urlopen(request, timeout=self.settings.request_timeout_sec)
        return urllib.request.urlopen(
            request,
            timeout=self.settings.request_timeout_sec,
            context=self.ssl_context,
        )

    @staticmethod
    def _coerce_vector(raw: List[object]) -> Optional[List[float]]:
        vector: List[float] = []
        for entry in raw:
            if not isinstance(entry, (float, int)):
                return None
            number = float(entry)
            if not math.isfinite(number):
                return None
            vector.append(number)
        return vector if vector else None

    @staticmethod
    def _valid_vector(vector: Optional[List[float]]) -> bool:
        if not vector:
            return False
        return all(math.isfinite(number) for number in vector)
