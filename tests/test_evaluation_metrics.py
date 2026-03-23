import unittest

import app.evaluation as evaluation


class _StubResult:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self):
        return self._payload


class _StubService:
    def __init__(self, settings):
        self.settings = settings

    def recommend(self, query: str, top_k: int = 5):
        if self.settings.semantic_weight == 0:
            ids = ["11", "12"]
        elif self.settings.enable_llm_rerank:
            ids = ["12", "11"]
        else:
            ids = ["12", "13"]
        shortlist = [
            {
                "product_id": product_id,
                "title": f"Product {product_id}",
                "price": 10.0 + idx,
                "rating": 4.0,
                "score": 0.5 - (idx * 0.1),
            }
            for idx, product_id in enumerate(ids[:top_k])
        ]
        best = shortlist[0] if shortlist else None
        return _StubResult(
            {
                "shortlist": shortlist,
                "best_value": best,
                "explanation_mode": "deterministic",
                "debug": {"rerank_used": bool(self.settings.enable_llm_rerank)},
            }
        )


class EvaluationMetricsTests(unittest.TestCase):
    def test_metrics_at_k_computation(self) -> None:
        metrics = evaluation._metrics_at_k(
            predicted_ids=["A", "B", "C"],
            qrels_for_query={"A": 3, "D": 2},
            k=2,
        )
        self.assertEqual(metrics["precision_at_k"], 0.5)
        self.assertEqual(metrics["recall_at_k"], 0.5)
        self.assertIsInstance(metrics["ndcg_at_k"], float)
        self.assertEqual(metrics["mrr_at_k"], 1.0)

    def test_run_evaluation_produces_all_modes(self) -> None:
        original_queries_loader = evaluation._load_benchmark_queries
        original_qrels_loader = evaluation._load_qrels
        try:
            evaluation._load_benchmark_queries = lambda: [
                evaluation.BenchmarkQuery(query_id="Q1", query="q1", intent="best_value", k=2),
                evaluation.BenchmarkQuery(query_id="Q2", query="q2", intent="best_value", k=2),
            ]
            evaluation._load_qrels = lambda: {
                "Q1": {"12": 3, "11": 2},
                "Q2": {"33": 3},
            }
            summary = evaluation.run_evaluation(service_factory=lambda settings: _StubService(settings))
        finally:
            evaluation._load_benchmark_queries = original_queries_loader
            evaluation._load_qrels = original_qrels_loader

        self.assertIn("modes", summary)
        modes = summary["modes"]
        self.assertIn("baseline_lexical", modes)
        self.assertIn("hybrid_semantic", modes)
        self.assertIn("hybrid_semantic_llm_rerank", modes)
        self.assertIn("aggregate_metrics", modes["baseline_lexical"])
        self.assertEqual(len(modes["baseline_lexical"]["queries"]), 2)


if __name__ == "__main__":
    unittest.main()
