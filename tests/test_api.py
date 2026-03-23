import unittest

try:
    from fastapi.testclient import TestClient
except (ModuleNotFoundError, RuntimeError):  # pragma: no cover - optional dependency in local CI/runtime
    TestClient = None

from app.types import ParsedQuery, RankedProduct, RecommendationResult


class _StubService:
    def recommend(self, query: str, top_k: int = 5, use_rapidapi: bool = False) -> RecommendationResult:
        parsed = ParsedQuery(
            query=query,
            normalized_query=query.lower(),
            category="electronics",
            budget=150.0,
            intent="best_value",
        )
        ranked = RankedProduct(
            product_id="11",
            title="Silicon Power SSD",
            category="electronics",
            price=109.0,
            rating=4.8,
            reviews=319,
            description="Fast SSD",
            source="catalog_csv",
            relevance=0.8,
            score=0.72,
            breakdown={"final_score": 0.72},
        )
        return RecommendationResult(
            parsed_query=parsed,
            candidates_considered=1,
            shortlist=[ranked],
            best_value=ranked,
            explanation="Silicon Power SSD is best for value [E1].",
            provider_used="ollama",
            source_trace=["catalog_csv"],
            explanation_mode="rag_grounded",
            citations=[
                {
                    "evidence_id": "E1",
                    "product_id": "11",
                    "field": "price",
                    "quote_or_value": "109.0",
                }
            ],
            grounding={
                "grounded_generation_used": True,
                "evidence_count": 1,
                "citation_count": 1,
                "fallback_reason": None,
            },
            debug={"ok": True},
        )


@unittest.skipUnless(TestClient is not None, "fastapi is not installed")
class ApiContractTests(unittest.TestCase):
    def test_recommend_response_includes_grounding_fields(self) -> None:
        from app import api as api_module

        original_service = api_module.RecommendationService
        try:
            api_module.RecommendationService = lambda settings: _StubService()  # type: ignore[assignment]
            app = api_module.create_app()
            client = TestClient(app)
            response = client.post(
                "/recommend",
                json={"query": "Best electronics under $150", "top_k": 3},
            )
        finally:
            api_module.RecommendationService = original_service

        self.assertEqual(response.status_code, 200)
        payload = response.json()

        self.assertIn("explanation", payload)
        self.assertIn("best_value", payload)
        self.assertIn("shortlist", payload)
        self.assertIn("explanation_mode", payload)
        self.assertIn("citations", payload)
        self.assertIn("grounding", payload)
        self.assertEqual(payload["explanation_mode"], "rag_grounded")
        self.assertEqual(len(payload["citations"]), 1)


if __name__ == "__main__":
    unittest.main()
