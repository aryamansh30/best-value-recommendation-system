import unittest

from app.config import Settings
from app.pipeline import RecommendationService
from app.types import RankedProduct, RawProduct


class _StubRetriever:
    def __init__(self, first_result, second_result=None):
        self.first_result = first_result
        self.second_result = second_result if second_result is not None else first_result
        self.calls = 0

    def retrieve(self, parsed_query, query_terms, limit=50, use_rapidapi=False):
        self.calls += 1
        if self.calls == 1:
            return self.first_result, ["catalog_csv"]
        return self.second_result, ["catalog_csv"]


class _StubGenAI:
    def __init__(self, synonyms, grounded_payload=None, grounded_exception: Exception | None = None):
        self.synonyms = synonyms
        self.grounded_payload = grounded_payload
        self.grounded_exception = grounded_exception
        self.expand_calls = 0
        self.grounded_calls = 0
        self.rewrite_calls = 0
        self.enabled = True

    def expand_synonyms(self, query, category, required_terms, max_terms=5):
        self.expand_calls += 1
        return self.synonyms

    def generate_grounded_explanation(self, query, intent, evidence, best_product=None):
        self.grounded_calls += 1
        if self.grounded_exception is not None:
            raise self.grounded_exception
        return self.grounded_payload

    def rewrite_explanation(self, explanation, context):
        self.rewrite_calls += 1
        return explanation


def _raw_product(
    product_id: str,
    title: str,
    category: str,
    price: float,
    rating: float,
    reviews: int,
    retrieval_score: float,
) -> RawProduct:
    return RawProduct(
        product_id=product_id,
        title=title,
        category=category,
        price=price,
        rating=rating,
        reviews=reviews,
        description=title,
        source="catalog_csv",
        retrieval_score=retrieval_score,
        discount=0.0,
    )


class PipelineFallbackTests(unittest.TestCase):
    @staticmethod
    def _grounded_payload(evidence_id: str, product_id: str, text: str = "Grounded explanation [E1]."):
        return {
            "explanation": text.replace("E1", evidence_id),
            "citations": [
                {
                    "evidence_id": evidence_id,
                    "product_id": product_id,
                    "field": "price",
                    "quote_or_value": "109.0",
                }
            ],
            "used_evidence_ids": [evidence_id],
            "limitations": "Based on available product catalog.",
        }

    @staticmethod
    def _ranked_product(product_id: str, title: str, description: str) -> RankedProduct:
        return RankedProduct(
            product_id=product_id,
            title=title,
            category="electronics",
            price=99.0,
            rating=4.7,
            reviews=250,
            description=description,
            source="fixture",
            relevance=0.8,
            score=0.77,
            breakdown={"final_score": 0.77},
        )

    def _service(self) -> RecommendationService:
        return RecommendationService(Settings())

    def test_build_evidence_pack_uses_ranked_products_with_stable_ids(self) -> None:
        ranked = [
            self._ranked_product(product_id="p1", title="A", description="A" * 20),
            self._ranked_product(product_id="p2", title="B", description="B" * 20),
        ]
        evidence = RecommendationService._build_evidence_pack(ranked, max_items=5)

        self.assertEqual([item["evidence_id"] for item in evidence], ["E1", "E2"])
        self.assertEqual(evidence[0]["product_id"], "p1")
        self.assertEqual(evidence[1]["product_id"], "p2")

    def test_uses_grounded_generation_when_payload_is_valid(self) -> None:
        service = self._service()
        product = _raw_product(
            product_id="11",
            title="Silicon Power SSD",
            category="electronics",
            price=109.0,
            rating=4.8,
            reviews=319,
            retrieval_score=0.8,
        )
        service.retriever = _StubRetriever(first_result=[product])
        genai = _StubGenAI(
            synonyms=["solid state drive"],
            grounded_payload=self._grounded_payload(evidence_id="E1", product_id="11"),
        )
        service.genai = genai

        result = service.recommend("Best electronics under $150", top_k=3)

        self.assertIsNotNone(result.best_value)
        self.assertTrue(result.parsed_query.used_genai)
        self.assertEqual(genai.expand_calls, 0)
        self.assertEqual(genai.grounded_calls, 1)
        self.assertEqual(result.explanation_mode, "rag_grounded")
        self.assertGreater(len(result.citations), 0)
        self.assertTrue(result.grounding.get("grounded_generation_used"))
        self.assertFalse(result.debug.get("genai_synonym_fallback_used"))

    def test_uses_genai_synonym_fallback_after_no_match(self) -> None:
        service = self._service()
        retry_product = _raw_product(
            product_id="18",
            title="Women's Solid Short Sleeve Boat Neck V",
            category="women's clothing",
            price=9.85,
            rating=4.7,
            reviews=130,
            retrieval_score=0.7,
        )
        service.retriever = _StubRetriever(first_result=[], second_result=[retry_product])
        genai = _StubGenAI(
            synonyms=["tee shirt"],
            grounded_payload=self._grounded_payload(evidence_id="E1", product_id="18"),
        )
        service.genai = genai

        result = service.recommend("Best t-shirt under $50", top_k=3)

        self.assertEqual(genai.expand_calls, 1)
        self.assertTrue(result.parsed_query.used_genai)
        self.assertTrue(result.debug.get("genai_synonym_fallback_attempted"))
        self.assertTrue(result.debug.get("genai_synonym_fallback_used"))
        self.assertEqual(result.debug.get("genai_synonyms"), ["tee shirt"])
        self.assertIsNotNone(result.best_value)
        self.assertEqual(result.explanation_mode, "rag_grounded")

    def test_grounded_generation_failure_falls_back_to_deterministic(self) -> None:
        service = self._service()
        product = _raw_product(
            product_id="11",
            title="Silicon Power SSD",
            category="electronics",
            price=109.0,
            rating=4.8,
            reviews=319,
            retrieval_score=0.8,
        )
        service.retriever = _StubRetriever(first_result=[product])
        service.genai = _StubGenAI(
            synonyms=[],
            grounded_exception=TimeoutError("llm timeout"),
        )

        result = service.recommend("Best electronics under $150", top_k=3)

        self.assertEqual(result.explanation_mode, "deterministic")
        self.assertEqual(result.citations, [])
        self.assertFalse(result.grounding.get("grounded_generation_used"))
        self.assertEqual(
            result.grounding.get("fallback_reason"),
            "invalid_or_unavailable_grounded_generation",
        )
        self.assertIn("deterministic", result.explanation.lower())

    def test_invalid_citation_payload_falls_back_without_leaking_text(self) -> None:
        service = self._service()
        product = _raw_product(
            product_id="11",
            title="Silicon Power SSD",
            category="electronics",
            price=109.0,
            rating=4.8,
            reviews=319,
            retrieval_score=0.8,
        )
        service.retriever = _StubRetriever(first_result=[product])
        invalid_payload = self._grounded_payload(
            evidence_id="E99",
            product_id="11",
            text="UNSAFE UNGROUNDED TEXT [E99].",
        )
        service.genai = _StubGenAI(synonyms=[], grounded_payload=invalid_payload)

        result = service.recommend("Best electronics under $150", top_k=3)

        self.assertEqual(result.explanation_mode, "deterministic")
        self.assertEqual(result.citations, [])
        self.assertFalse(result.grounding.get("grounded_generation_used"))
        self.assertEqual(
            result.grounding.get("fallback_reason"),
            "invalid_or_unavailable_grounded_generation",
        )
        self.assertNotIn("UNSAFE UNGROUNDED TEXT", result.explanation)


if __name__ == "__main__":
    unittest.main()
