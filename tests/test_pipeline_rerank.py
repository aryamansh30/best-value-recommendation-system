import unittest

from app.config import Settings
from app.pipeline import RecommendationService
from app.types import RawProduct


def _raw_product(
    product_id: str,
    title: str,
    price: float,
    rating: float,
    reviews: int,
    retrieval_score: float,
) -> RawProduct:
    return RawProduct(
        product_id=product_id,
        title=title,
        category="electronics",
        price=price,
        rating=rating,
        reviews=reviews,
        description=title,
        source="fixture",
        retrieval_score=retrieval_score,
        lexical_score=retrieval_score,
        semantic_score=0.0,
        fused_score=retrieval_score,
        discount=0.0,
    )


class _StubRetriever:
    def __init__(self, products):
        self.products = products
        self.last_debug = {
            "lexical_weight": 1.0,
            "semantic_weight": 0.0,
            "semantic_fallback_reason": "semantic_weight_disabled",
            "retrieval_breakdown": [],
        }

    def retrieve(self, parsed_query, query_terms, limit=50, use_rapidapi=False):
        return self.products, ["fixture"]


class _StubGenAI:
    def __init__(self, rerank_payload=None):
        self.rerank_payload = rerank_payload
        self.rerank_calls = 0
        self.enabled = True

    def expand_synonyms(self, query, category, required_terms, max_terms=5):
        return []

    def rerank_candidates(self, query, intent, candidates):
        self.rerank_calls += 1
        return self.rerank_payload

    def generate_grounded_explanation(self, query, intent, evidence, best_product=None):
        first = evidence[0]
        return {
            "explanation": f"Grounded recommendation [{first['evidence_id']}].",
            "citations": [
                {
                    "evidence_id": first["evidence_id"],
                    "product_id": first["product_id"],
                    "field": "title",
                    "quote_or_value": first["title"],
                }
            ],
            "used_evidence_ids": [first["evidence_id"]],
            "limitations": "catalog-only",
        }


class PipelineRerankTests(unittest.TestCase):
    def _products(self):
        return [
            _raw_product(
                product_id="11",
                title="Silicon Power SSD",
                price=109.0,
                rating=4.8,
                reviews=319,
                retrieval_score=0.8,
            ),
            _raw_product(
                product_id="12",
                title="WD Gaming Drive",
                price=114.0,
                rating=4.8,
                reviews=400,
                retrieval_score=0.79,
            ),
        ]

    def test_llm_rerank_applies_when_enabled(self) -> None:
        settings = Settings(enable_llm_rerank=True, llm_rerank_top_n=2, llm_rerank_weight=1.0)
        service = RecommendationService(settings)
        service.retriever = _StubRetriever(products=self._products())
        service.genai = _StubGenAI(
            rerank_payload={
                "ordered_product_ids": ["12", "11"],
                "rationale": "12 better aligns with intent.",
                "citations": ["reviews", "rating"],
            }
        )

        result = service.recommend("Best electronics under $150", top_k=2)

        self.assertEqual(result.best_value.product_id, "12")
        self.assertTrue(result.debug.get("rerank_attempted"))
        self.assertTrue(result.debug.get("rerank_used"))
        self.assertIsNone(result.debug.get("rerank_fallback_reason"))

    def test_llm_rerank_does_not_run_when_disabled(self) -> None:
        settings = Settings(enable_llm_rerank=False, llm_rerank_top_n=2, llm_rerank_weight=1.0)
        service = RecommendationService(settings)
        service.retriever = _StubRetriever(products=self._products())
        stub_genai = _StubGenAI(
            rerank_payload={
                "ordered_product_ids": ["12", "11"],
                "rationale": "12 better aligns with intent.",
            }
        )
        service.genai = stub_genai

        result = service.recommend("Best electronics under $150", top_k=2)

        self.assertEqual(result.best_value.product_id, "11")
        self.assertEqual(stub_genai.rerank_calls, 0)
        self.assertFalse(result.debug.get("rerank_attempted"))
        self.assertFalse(result.debug.get("rerank_used"))

    def test_invalid_rerank_payload_falls_back_to_deterministic_order(self) -> None:
        settings = Settings(enable_llm_rerank=True, llm_rerank_top_n=2, llm_rerank_weight=1.0)
        service = RecommendationService(settings)
        service.retriever = _StubRetriever(products=self._products())
        service.genai = _StubGenAI(
            rerank_payload={
                "ordered_product_ids": ["99", "11"],
                "rationale": "invalid ids",
            }
        )

        result = service.recommend("Best electronics under $150", top_k=2)

        self.assertEqual(result.best_value.product_id, "11")
        self.assertTrue(result.debug.get("rerank_attempted"))
        self.assertFalse(result.debug.get("rerank_used"))
        self.assertEqual(result.debug.get("rerank_fallback_reason"), "rerank_invalid_order")


if __name__ == "__main__":
    unittest.main()
