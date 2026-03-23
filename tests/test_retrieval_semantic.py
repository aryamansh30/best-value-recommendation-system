import unittest

from app.config import Settings
from app.retrieval import ProductRetriever
from app.types import ParsedQuery, RawProduct


class _StubEmbeddingClient:
    def __init__(self, vectors: dict[str, list[float]] | None = None, available: bool = True):
        self.vectors = vectors or {}
        self._available = available
        self.provider = "openai"
        self.model = "test-embed"

    def available(self) -> bool:
        return self._available

    def embed_text(self, text: str, cache_id: str):
        if cache_id.startswith("query:"):
            return self.vectors.get("query")
        product_id = cache_id.split(":", 1)[-1]
        return self.vectors.get(product_id)


def _product(product_id: str, title: str) -> RawProduct:
    return RawProduct(
        product_id=product_id,
        title=title,
        category="electronics",
        price=99.0,
        rating=4.5,
        reviews=150,
        description=title,
        source="fixture",
        discount=0.0,
    )


class SemanticRetrievalTests(unittest.TestCase):
    def _retriever(self) -> ProductRetriever:
        settings = Settings(
            semantic_weight=0.65,
            lexical_weight=0.35,
            openai_api_key="test-key",
        )
        retriever = ProductRetriever(settings)
        retriever._fetch_rapidapi = lambda query: []  # type: ignore[method-assign]
        return retriever

    def _parsed(self) -> ParsedQuery:
        return ParsedQuery(
            query="best ssd under $150",
            normalized_query="best ssd under $150",
            category="electronics",
            intent="best_value",
        )

    def test_hybrid_scoring_uses_semantic_signal(self) -> None:
        retriever = self._retriever()
        products = [
            _product("11", "Portable Solid State Drive"),
            _product("22", "Wired Mouse"),
        ]
        retriever._fetch_fakestore = lambda parsed_query, limit: products  # type: ignore[method-assign]
        retriever._embedding_client = _StubEmbeddingClient(
            vectors={
                "query": [1.0, 0.0],
                "11": [1.0, 0.0],
                "22": [0.0, 1.0],
            }
        )

        ranked, _ = retriever.retrieve(self._parsed(), query_terms=["ssd"], limit=10)

        self.assertEqual(ranked[0].product_id, "11")
        self.assertGreater(ranked[0].semantic_score, ranked[1].semantic_score)
        self.assertIsNone(retriever.last_debug.get("semantic_fallback_reason"))
        self.assertFalse(retriever.last_debug.get("force_lexical_only"))

    def test_unavailable_embeddings_force_lexical_only(self) -> None:
        retriever = self._retriever()
        products = [_product("11", "Portable Solid State Drive")]
        retriever._fetch_fakestore = lambda parsed_query, limit: products  # type: ignore[method-assign]
        retriever._embedding_client = _StubEmbeddingClient(available=False)

        ranked, _ = retriever.retrieve(self._parsed(), query_terms=["ssd"], limit=10)

        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0].semantic_score, 0.0)
        self.assertTrue(retriever.last_debug.get("force_lexical_only"))
        self.assertEqual(
            retriever.last_debug.get("semantic_fallback_reason"),
            "embedding_provider_unavailable",
        )

    def test_invalid_embedding_shapes_trigger_fallback(self) -> None:
        retriever = self._retriever()
        products = [_product("11", "Portable Solid State Drive")]
        retriever._fetch_fakestore = lambda parsed_query, limit: products  # type: ignore[method-assign]
        retriever._embedding_client = _StubEmbeddingClient(
            vectors={
                "query": [1.0, 0.0],
                "11": [1.0, 0.0, 0.0],
            }
        )

        retriever.retrieve(self._parsed(), query_terms=["ssd"], limit=10)

        self.assertEqual(
            retriever.last_debug.get("semantic_fallback_reason"),
            "no_valid_product_embeddings",
        )
        self.assertTrue(retriever.last_debug.get("force_lexical_only"))


if __name__ == "__main__":
    unittest.main()
