import unittest
from pathlib import Path

from app.config import Settings
from app.retrieval import ProductRetriever
from app.types import ParsedQuery


class RetrievalCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        settings = Settings(catalog_csv_path=str(project_root / "data" / "enhanced_fakestore_products.csv"))
        self.retriever = ProductRetriever(settings)

    def _parsed(self, category: str | None, intent: str = "best_value") -> ParsedQuery:
        return ParsedQuery(
            query="q",
            normalized_query="q",
            category=category,
            intent=intent,
        )

    def test_clothing_maps_to_mens_and_womens_rows(self) -> None:
        products = self.retriever._fetch_fakestore(
            parsed_query=self._parsed(category="clothing"),
            limit=100,
        )
        categories = {item.category for item in products}
        self.assertTrue(categories.issubset({"men's clothing", "women's clothing"}))
        self.assertGreater(len(products), 0)

    def test_cheapest_intent_sorts_by_price_ascending(self) -> None:
        products = self.retriever._fetch_fakestore(
            parsed_query=self._parsed(category="electronics", intent="cheapest"),
            limit=6,
        )
        prices = [float(item.price) for item in products]
        self.assertEqual(prices, sorted(prices))

    def test_premium_intent_sorts_by_price_descending(self) -> None:
        products = self.retriever._fetch_fakestore(
            parsed_query=self._parsed(category="electronics", intent="premium"),
            limit=6,
        )
        prices = [float(item.price) for item in products]
        self.assertEqual(prices, sorted(prices, reverse=True))

    def test_protein_bars_category_maps_to_grocery(self) -> None:
        products = self.retriever._fetch_fakestore(
            parsed_query=self._parsed(category="protein bars", intent="best_value"),
            limit=30,
        )
        self.assertGreater(len(products), 0)
        self.assertTrue(all(item.category == "grocery" for item in products))


if __name__ == "__main__":
    unittest.main()
