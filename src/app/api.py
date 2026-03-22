from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import Settings
from .genai import GenAIError
from .pipeline import RecommendationService


class RecommendRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    use_rapidapi: bool = False


def create_app() -> FastAPI:
    settings = Settings.from_env()
    service = RecommendationService(settings)

    app = FastAPI(
        title="Best-Value Recommendation API",
        version="1.0.0",
        description="Deterministic recommendation core with selective GenAI synonym fallback for no-match retrieval.",
    )

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "genai_enabled": True,
            "provider": settings.active_provider(),
        }

    @app.post("/recommend")
    def recommend(request: RecommendRequest) -> dict:
        try:
            result = service.recommend(
                query=request.query,
                top_k=request.top_k,
                use_rapidapi=request.use_rapidapi,
            )
            return result.to_dict()
        except (GenAIError, RuntimeError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


app = create_app()
