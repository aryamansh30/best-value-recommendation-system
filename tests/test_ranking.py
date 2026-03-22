import unittest

from app.normalization import normalize_products
from app.ranking import rank_products
from tests.fixtures import SAMPLE_RAW_PRODUCTS


class RankingTests(unittest.TestCase):
    def test_cheapest_intent_prefers_lower_price(self) -> None:
        products = normalize_products(SAMPLE_RAW_PRODUCTS)
        ranked = rank_products(products, intent="cheapest", top_k=4)
        self.assertEqual(ranked[0].title, "Protein Snack Bars 12 Pack")

    def test_best_value_includes_breakdown(self) -> None:
        products = normalize_products(SAMPLE_RAW_PRODUCTS)
        ranked = rank_products(products, intent="best_value", top_k=3)
        self.assertIn("final_score", ranked[0].breakdown)
        self.assertGreaterEqual(ranked[0].score, ranked[-1].score)


if __name__ == "__main__":
    unittest.main()
