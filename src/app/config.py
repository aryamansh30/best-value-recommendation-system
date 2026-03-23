from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class Settings:
    fakestore_url: str = "https://fakestoreapi.com/products"
    catalog_csv_path: str = "data/enhanced_fakestore_products.csv"
    request_timeout_sec: int = 15
    ssl_ca_bundle: str = ""


    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    embedding_provider: str = "local"
    openai_embedding_model: str = "text-embedding-3-small"
    ollama_embedding_model: str = "nomic-embed-text:latest"
    local_embedding_model: str = "local-hash-v1"
    embedding_local_fallback: bool = True
    embedding_cache_path: str = ".cache/embeddings.json"
    semantic_weight: float = 0.65
    lexical_weight: float = 0.35
    enable_llm_rerank: bool = False
    llm_rerank_top_n: int = 8
    llm_rerank_weight: float = 0.30


    use_rapidapi: bool = False
    rapidapi_url: str = "https://real-time-amazon-data.p.rapidapi.com/search"
    rapidapi_host: str = "real-time-amazon-data.p.rapidapi.com"
    rapidapi_key: str = ""
    rapidapi_region: str = "US"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            fakestore_url=os.getenv("FAKESTORE_URL", "https://fakestoreapi.com/products"),
            catalog_csv_path=os.getenv("CATALOG_CSV_PATH", "data/enhanced_fakestore_products.csv"),
            request_timeout_sec=int(os.getenv("REQUEST_TIMEOUT_SEC", "15")),
            ssl_ca_bundle=os.getenv("SSL_CA_BUNDLE", "").strip(),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral:latest"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").strip().lower(),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            ollama_embedding_model=os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text:latest"),
            local_embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", "local-hash-v1"),
            embedding_local_fallback=os.getenv("EMBEDDING_LOCAL_FALLBACK", "true").strip().lower()
            in {"1", "true", "yes", "on"},
            embedding_cache_path=os.getenv("EMBEDDING_CACHE_PATH", ".cache/embeddings.json"),
            semantic_weight=float(os.getenv("SEMANTIC_WEIGHT", "0.65")),
            lexical_weight=float(os.getenv("LEXICAL_WEIGHT", "0.35")),
            enable_llm_rerank=os.getenv("ENABLE_LLM_RERANK", "").strip().lower()
            in {"1", "true", "yes", "on"},
            llm_rerank_top_n=max(1, int(os.getenv("LLM_RERANK_TOP_N", "8"))),
            llm_rerank_weight=float(os.getenv("LLM_RERANK_WEIGHT", "0.30")),
            use_rapidapi=os.getenv("USE_RAPIDAPI", "").strip().lower() in {"1", "true", "yes", "on"},
            rapidapi_url=os.getenv(
                "RAPIDAPI_URL",
                "https://real-time-amazon-data.p.rapidapi.com/search",
            ),
            rapidapi_host=os.getenv(
                "RAPIDAPI_HOST",
                "real-time-amazon-data.p.rapidapi.com",
            ),
            rapidapi_key=os.getenv("RAPIDAPI_KEY", ""),
            rapidapi_region=os.getenv("RAPIDAPI_REGION", "US"),
        )

    def active_provider(self) -> str:
        return self.llm_provider
