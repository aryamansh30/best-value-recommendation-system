import unittest

from app.config import Settings
from app.retrieval import ProductRetriever
from app.types import ParsedQuery


class RetrievalUrlBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retriever = ProductRetriever(Settings())

    def _parsed(self, category: str | None, intent: str = "best_value") -> ParsedQuery:
        return ParsedQuery(
            query="q",
            normalized_query="q",
            category=category,
            intent=intent,
        )

    def test_global_products_route_uses_limit_and_sort(self) -> None:
        urls = self.retriever._build_fakestore_request_urls(
            parsed_query=self._parsed(category=None, intent="cheapest"),
            limit=7,
        )
        self.assertEqual(len(urls), 1)
        self.assertIn("/products?", urls[0])
        self.assertIn("limit=7", urls[0])
        self.assertIn("sort=asc", urls[0])

    def test_clothing_maps_to_two_category_routes(self) -> None:
        urls = self.retriever._build_fakestore_request_urls(
            parsed_query=self._parsed(category="clothing"),
            limit=20,
        )
        self.assertEqual(len(urls), 2)
        self.assertTrue(any("/products/category/men%27s%20clothing" in url for url in urls))
        self.assertTrue(any("/products/category/women%27s%20clothing" in url for url in urls))

    def test_electronics_maps_to_category_route(self) -> None:
        urls = self.retriever._build_fakestore_request_urls(
            parsed_query=self._parsed(category="electronics", intent="premium"),
            limit=5,
        )
        self.assertEqual(len(urls), 1)
        self.assertIn("/products/category/electronics", urls[0])
        self.assertIn("sort=desc", urls[0])


if __name__ == "__main__":
    unittest.main()
