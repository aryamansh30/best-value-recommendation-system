import unittest

from app.config import Settings
from app.genai import GenAIError
from app.pipeline import RecommendationService


class PipelineIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings.from_env()
        self.settings.use_rapidapi = False
        self.service = RecommendationService(self.settings)

    def _recommend_or_skip(self, query: str, top_k: int = 5):
        try:
            return self.service.recommend(query=query, top_k=top_k)
        except (GenAIError, RuntimeError) as exc:
            self.skipTest(f"Dependency unavailable for integration test: {exc}")

    def _require_dependencies(self) -> None:
        self._recommend_or_skip(query="Best t-shirt under $50", top_k=1)

    def test_end_to_end_uses_catalog_source_with_grounding_metadata(self) -> None:
        self._require_dependencies()
        result = self._recommend_or_skip("Best electronics under $150", top_k=5)

        self.assertIn("catalog_csv", result.source_trace)
        self.assertFalse(result.debug.get("genai_synonym_fallback_used"))
        self.assertIsNotNone(result.best_value)
        self.assertIn(result.explanation_mode, {"deterministic", "rag_grounded"})
        self.assertIn("grounded_generation_used", result.grounding)

    def test_evaluation_queries_align_with_current_dataset(self) -> None:
        self._require_dependencies()
        expectations = [
            ("Best electronics under $150", True),
            ("Cheapest jewelry under $20", True),
            ("Best t-shirt under $50", True),
            ("Best laptop under $150", False),
        ]

        for query, should_have_result in expectations:
            with self.subTest(query=query):
                result = self._recommend_or_skip(query=query, top_k=5)
                if should_have_result:
                    self.assertIsNotNone(result.best_value)
                    self.assertGreater(len(result.shortlist), 0)
                    self.assertFalse(result.explanation.strip().startswith("{"))
                    self.assertIn(result.explanation_mode, {"deterministic", "rag_grounded"})
                else:
                    self.assertIsNone(result.best_value)
                    self.assertEqual(len(result.shortlist), 0)
                    self.assertIn("no product found", result.explanation.lower())


if __name__ == "__main__":
    unittest.main()
