import unittest

from app.normalization import normalize_products
from app.ranking import rank_products
from app.types import RawProduct


SAMPLE_RAW_PRODUCTS = [
    RawProduct(
        product_id="1",
        title="Wireless Over-Ear Headphones",
        category="electronics",
        price=99.99,
        rating=4.4,
        reviews=240,
        description="Bluetooth wireless headphones with noise isolation",
        source="fixture",
        retrieval_score=0.92,
        discount=15.0,
    ),
    RawProduct(
        product_id="2",
        title="Budget Wired Earbuds",
        category="electronics",
        price=24.99,
        rating=3.8,
        reviews=120,
        description="Affordable earbuds for daily use",
        source="fixture",
        retrieval_score=0.71,
        discount=0.0,
    ),
    RawProduct(
        product_id="3",
        title="Mechanical Gaming Keyboard",
        category="electronics",
        price=79.0,
        rating=4.6,
        reviews=310,
        description="RGB mechanical keyboard for gaming and typing",
        source="fixture",
        retrieval_score=0.88,
        discount=10.0,
    ),
    RawProduct(
        product_id="4",
        title="Protein Snack Bars 12 Pack",
        category="groceries",
        price=18.5,
        rating=4.2,
        reviews=95,
        description="High protein snack bars",
        source="fixture",
        retrieval_score=0.87,
        discount=5.0,
    ),
    RawProduct(
        product_id="5",
        title="Mens Casual Premium Slim Fit T-Shirts",
        category="men's clothing",
        price=22.3,
        rating=4.1,
        reviews=259,
        description="Slim-fitting style long sleeve t-shirt for casual wear.",
        source="fixture",
        retrieval_score=0.9,
        discount=0.0,
    ),
]


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
