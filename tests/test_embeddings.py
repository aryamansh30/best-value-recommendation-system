import tempfile
import unittest
from pathlib import Path

from app.config import Settings
from app.embeddings import EmbeddingClient, cosine_similarity, normalize_weight_pair


class _StubEmbeddingClient(EmbeddingClient):
    def __init__(self, settings: Settings):
        super().__init__(settings=settings, ssl_context=None)
        self.calls = 0

    def _embed_openai(self, text: str):
        self.calls += 1
        return [0.1, 0.2, 0.3]


class EmbeddingMathTests(unittest.TestCase):
    def test_cosine_similarity_handles_basic_vectors(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [0.0, 1.0]), 0.0)
        self.assertEqual(cosine_similarity([1.0], [1.0, 2.0]), 0.0)

    def test_normalize_weight_pair(self) -> None:
        lexical, semantic = normalize_weight_pair(0.35, 0.65)
        self.assertAlmostEqual(lexical, 0.35)
        self.assertAlmostEqual(semantic, 0.65)

        lexical, semantic = normalize_weight_pair(0.0, 0.0)
        self.assertAlmostEqual(lexical, 1.0)
        self.assertAlmostEqual(semantic, 0.0)


class EmbeddingCacheTests(unittest.TestCase):
    def test_cache_hit_and_text_hash_invalidation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "embeddings.json"
            settings = Settings(
                openai_api_key="test-key",
                embedding_provider="openai",
                embedding_cache_path=str(cache_path),
            )
            client = _StubEmbeddingClient(settings=settings)

            first = client.embed_text("hello world", cache_id="query:q1")
            second = client.embed_text("hello world", cache_id="query:q1")
            third = client.embed_text("hello world v2", cache_id="query:q1")

            self.assertIsNotNone(first)
            self.assertEqual(first, second)
            self.assertEqual(client.calls, 2)
            self.assertIsNotNone(third)

    def test_cache_key_separates_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "embeddings.json"
            settings_a = Settings(
                openai_api_key="test-key",
                embedding_provider="openai",
                openai_embedding_model="model-a",
                embedding_cache_path=str(cache_path),
            )
            settings_b = Settings(
                openai_api_key="test-key",
                embedding_provider="openai",
                openai_embedding_model="model-b",
                embedding_cache_path=str(cache_path),
            )
            client_a = _StubEmbeddingClient(settings=settings_a)
            client_b = _StubEmbeddingClient(settings=settings_b)

            client_a.embed_text("shared text", cache_id="product:1")
            client_b.embed_text("shared text", cache_id="product:1")
            self.assertEqual(client_a.calls, 1)
            self.assertEqual(client_b.calls, 1)


if __name__ == "__main__":
    unittest.main()
