import unittest

from app.config import Settings
from app.genai import GenAIError
from app.pipeline import RecommendationService


class PipelineLiveApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = Settings.from_env()
        self.settings.use_rapidapi = False
        self.service = RecommendationService(self.settings)

    def _recommend_or_skip(self, query: str, top_k: int = 5):
        try:
            return self.service.recommend(query=query, top_k=top_k)
        except (GenAIError, RuntimeError) as exc:
            self.skipTest(f"Live dependency unavailable for integration test: {exc}")

    def _require_live_dependencies(self) -> None:
        self._recommend_or_skip(query="Best t-shirt under $50", top_k=1)

    def test_end_to_end_uses_live_fakestore_deterministically_by_default(self) -> None:
        self._require_live_dependencies()
        result = self._recommend_or_skip("Best electronics under $150", top_k=5)

        self.assertIn("fakestore", result.source_trace)
        self.assertNotIn("fakestore_snapshot", result.source_trace)
        self.assertFalse(result.parsed_query.used_genai)
        self.assertFalse(result.debug.get("genai_synonym_fallback_used"))
        self.assertIsNotNone(result.best_value)

    def test_evaluation_queries_align_with_current_dataset(self) -> None:
        self._require_live_dependencies()
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
                else:
                    self.assertIsNone(result.best_value)
                    self.assertEqual(len(result.shortlist), 0)
                    self.assertIn("no product found", result.explanation.lower())


if __name__ == "__main__":
    unittest.main()
