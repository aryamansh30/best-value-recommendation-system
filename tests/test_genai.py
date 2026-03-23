import unittest

from app.config import Settings
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


class _StubGroundedGenAIClient(GenAIClient):
    def __init__(self, response: str):
        super().__init__(Settings())
        self._response = response

    def _chat(self, messages, force_json=False):
        return self._response


class GenAIGroundedExplanationTests(unittest.TestCase):
    @staticmethod
    def _evidence():
        return [
            {
                "evidence_id": "E1",
                "product_id": "11",
                "title": "Silicon Power SSD",
                "price": 109.0,
                "rating": 4.8,
                "reviews": 319,
                "source": "catalog_csv",
                "description_snippet": "Fast SSD for gaming workloads.",
            },
            {
                "evidence_id": "E2",
                "product_id": "12",
                "title": "WD Gaming Drive",
                "price": 114.0,
                "rating": 4.8,
                "reviews": 400,
                "source": "catalog_csv",
                "description_snippet": "Portable drive for consoles.",
            },
        ]

    def test_grounded_explanation_accepts_valid_json_with_citations(self) -> None:
        payload = (
            '{"explanation":"Product A is stronger on price and reviews [E1] while Product B is a close '
            'alternative [E2].","citations":[{"evidence_id":"E1","product_id":"11","field":"price",'
            '"quote_or_value":"109.0"},{"evidence_id":"E2","product_id":"12","field":"reviews",'
            '"quote_or_value":"400"}],"used_evidence_ids":["E1","E2"],"limitations":"Based only on available '
            'catalog data."}'
        )
        client = _StubGroundedGenAIClient(response=payload)
        grounded = client.generate_grounded_explanation(
            query="Best electronics under $150",
            intent="best_value",
            evidence=self._evidence(),
            best_product={"product_id": "11", "title": "Silicon Power SSD"},
        )
        self.assertIsNotNone(grounded)
        assert grounded is not None
        self.assertEqual(len(grounded["citations"]), 2)
        self.assertEqual(grounded["used_evidence_ids"], ["E1", "E2"])

    def test_grounded_explanation_uses_deterministic_fallback_for_malformed_json(self) -> None:
        client = _StubGroundedGenAIClient(response="this is not json")
        grounded = client.generate_grounded_explanation(
            query="Best electronics under $150",
            intent="best_value",
            evidence=self._evidence(),
        )
        self.assertIsNotNone(grounded)
        assert grounded is not None
        self.assertGreater(len(grounded["citations"]), 0)
        self.assertIn("[E1]", grounded["explanation"])


class GenAIRerankValidationTests(unittest.TestCase):
    def test_rerank_payload_validator_accepts_valid_permutation(self) -> None:
        payload = {
            "ordered_product_ids": ["11", "22", "33"],
            "rationale": "11 has the strongest balance.",
            "citations": ["price", "rating"],
        }
        validated = GenAIClient._validate_rerank_payload(payload, candidate_ids=["11", "22", "33"])
        self.assertIsNotNone(validated)
        assert validated is not None
        self.assertEqual(validated["ordered_product_ids"], ["11", "22", "33"])

    def test_rerank_payload_validator_rejects_unknown_ids(self) -> None:
        payload = {
            "ordered_product_ids": ["11", "22", "99"],
        }
        validated = GenAIClient._validate_rerank_payload(payload, candidate_ids=["11", "22", "33"])
        self.assertIsNone(validated)

    def test_rerank_payload_validator_rejects_duplicates(self) -> None:
        payload = {
            "ordered_product_ids": ["11", "11", "33"],
        }
        validated = GenAIClient._validate_rerank_payload(payload, candidate_ids=["11", "22", "33"])
        self.assertIsNone(validated)

    def test_grounded_explanation_uses_deterministic_fallback_for_invalid_evidence_reference(self) -> None:
        payload = (
            '{"explanation":"This product wins on rating [E99].","citations":[{"evidence_id":"E99",'
            '"product_id":"11","field":"rating","quote_or_value":"4.8"}],"used_evidence_ids":["E99"],'
            '"limitations":"n/a"}'
        )
        client = _StubGroundedGenAIClient(response=payload)
        grounded = client.generate_grounded_explanation(
            query="Best electronics under $150",
            intent="best_value",
            evidence=GenAIGroundedExplanationTests._evidence(),
        )
        self.assertIsNotNone(grounded)
        assert grounded is not None
        self.assertGreater(len(grounded["citations"]), 0)
        self.assertIn("[E1]", grounded["explanation"])


if __name__ == "__main__":
    unittest.main()
