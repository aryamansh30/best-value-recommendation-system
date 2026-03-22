import unittest

from app.query_parser import parse_query


class QueryParserTests(unittest.TestCase):
    def test_parse_budget_intent_and_category(self) -> None:
        parsed = parse_query("Best wireless headphones under $150")
        self.assertEqual(parsed.category, "headphones")
        self.assertEqual(parsed.intent, "best_value")
        self.assertEqual(parsed.budget, 150.0)
        self.assertEqual(parsed.filters.get("type"), "wireless")

    def test_parse_cheapest_query(self) -> None:
        parsed = parse_query("Cheapest protein snack bars")
        self.assertEqual(parsed.category, "protein bars")
        self.assertEqual(parsed.intent, "cheapest")

    def test_parse_general_query(self) -> None:
        parsed = parse_query("Need something for music")
        self.assertEqual(parsed.intent, "general_search")
        self.assertIsNone(parsed.budget)

    def test_parse_tshirt_budget_query(self) -> None:
        parsed = parse_query("Best t-shirt under $50")
        self.assertEqual(parsed.category, "clothing")
        self.assertEqual(parsed.budget, 50.0)


if __name__ == "__main__":
    unittest.main()
