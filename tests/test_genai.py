import unittest

from app.genai import GenAIClient


class GenAITextExtractionTests(unittest.TestCase):
    def test_extracts_recommendation_field_from_json(self) -> None:
        payload = '{"recommendation":"Use product A."}'
        text = GenAIClient._extract_explanation_text(payload)
        self.assertEqual(text, "Use product A.")

    def test_returns_plain_text_as_is(self) -> None:
        payload = "Use product B because it has better rating."
        text = GenAIClient._extract_explanation_text(payload)
        self.assertEqual(text, payload)


if __name__ == "__main__":
    unittest.main()
