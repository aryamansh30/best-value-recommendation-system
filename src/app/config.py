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
    request_timeout_sec: int = 15
    ssl_ca_bundle: str = ""

    # Optional model provider switching (not core)
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "mistral:latest"
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"

    # Optional RapidAPI source (not core)
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
            request_timeout_sec=int(os.getenv("REQUEST_TIMEOUT_SEC", "15")),
            ssl_ca_bundle=os.getenv("SSL_CA_BUNDLE", "").strip(),
            llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_model=os.getenv("OLLAMA_MODEL", "mistral:latest"),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
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
