import unittest

from app.config import Settings
from app.pipeline import RecommendationService
from app.types import RawProduct


class _StubRetriever:
    def __init__(self, first_result, second_result=None):
        self.first_result = first_result
        self.second_result = second_result if second_result is not None else first_result
        self.calls = 0

    def retrieve(self, parsed_query, query_terms, limit=50, use_rapidapi=False):
        self.calls += 1
        if self.calls == 1:
            return self.first_result, ["fakestore"]
        return self.second_result, ["fakestore"]


class _StubGenAI:
    def __init__(self, synonyms):
        self.synonyms = synonyms
        self.expand_calls = 0
        self.rewrite_calls = 0
        self.enabled = True

    def expand_synonyms(self, query, category, required_terms, max_terms=5):
        self.expand_calls += 1
        return self.synonyms

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
        source="fakestore",
        retrieval_score=retrieval_score,
        discount=0.0,
    )


class PipelineFallbackTests(unittest.TestCase):
    def _service(self) -> RecommendationService:
        return RecommendationService(Settings())

    def test_does_not_use_genai_when_deterministic_retrieval_succeeds(self) -> None:
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
        genai = _StubGenAI(synonyms=["solid state drive"])
        service.genai = genai

        result = service.recommend("Best electronics under $150", top_k=3)

        self.assertIsNotNone(result.best_value)
        self.assertFalse(result.parsed_query.used_genai)
        self.assertEqual(genai.expand_calls, 0)
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
        genai = _StubGenAI(synonyms=["tee shirt"])
        service.genai = genai

        result = service.recommend("Best t-shirt under $50", top_k=3)

        self.assertEqual(genai.expand_calls, 1)
        self.assertTrue(result.parsed_query.used_genai)
        self.assertTrue(result.debug.get("genai_synonym_fallback_attempted"))
        self.assertTrue(result.debug.get("genai_synonym_fallback_used"))
        self.assertEqual(result.debug.get("genai_synonyms"), ["tee shirt"])
        self.assertIsNotNone(result.best_value)


if __name__ == "__main__":
    unittest.main()
