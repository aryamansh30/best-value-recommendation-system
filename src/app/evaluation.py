from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .config import Settings
from .genai import GenAIError
from .pipeline import RecommendationService

RUN_MODES = (
    "baseline_lexical",
    "hybrid_semantic",
    "hybrid_semantic_llm_rerank",
)


@dataclass
class BenchmarkQuery:
    query_id: str
    query: str
    intent: str = "general_search"
    k: int = 5


def _root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def _benchmark_queries_path() -> Path:
    return _root_dir() / "evaluation" / "benchmark_queries.jsonl"


def _qrels_path() -> Path:
    return _root_dir() / "evaluation" / "qrels.jsonl"


def _load_benchmark_queries() -> List[BenchmarkQuery]:
    path = _benchmark_queries_path()
    if not path.exists():
        # Fallback to legacy queries list for compatibility.
        return [
            BenchmarkQuery(query_id="Q1", query="Best electronics under $150", intent="best_value", k=5),
            BenchmarkQuery(query_id="Q2", query="Cheapest jewelry under $20", intent="cheapest", k=5),
            BenchmarkQuery(query_id="Q3", query="Best t-shirt under $50", intent="best_value", k=5),
            BenchmarkQuery(query_id="Q4", query="Best laptop under $150", intent="best_value", k=5),
        ]

    queries: List[BenchmarkQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        query_id = str(payload.get("query_id", "")).strip()
        query = str(payload.get("query", "")).strip()
        if not query_id or not query:
            continue
        intent = str(payload.get("intent", "general_search")).strip() or "general_search"
        try:
            k = max(1, int(payload.get("k", 5)))
        except (TypeError, ValueError):
            k = 5
        queries.append(BenchmarkQuery(query_id=query_id, query=query, intent=intent, k=k))
    return queries


def _load_qrels() -> Dict[str, Dict[str, int]]:
    path = _qrels_path()
    if not path.exists():
        return {}

    qrels: Dict[str, Dict[str, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        query_id = str(payload.get("query_id", "")).strip()
        product_id = str(payload.get("product_id", "")).strip()
        if not query_id or not product_id:
            continue
        try:
            relevance = int(payload.get("relevance", 0))
        except (TypeError, ValueError):
            continue
        relevance = max(0, min(relevance, 3))
        qrels.setdefault(query_id, {})[product_id] = relevance
    return qrels


def _dcg(relevances: List[int]) -> float:
    total = 0.0
    for rank_idx, rel in enumerate(relevances, start=1):
        if rel <= 0:
            continue
        total += (2**rel - 1) / math.log2(rank_idx + 1)
    return total


def _metrics_at_k(
    predicted_ids: List[str],
    qrels_for_query: Optional[Dict[str, int]],
    k: int,
) -> Dict[str, Optional[float]]:
    if not qrels_for_query:
        return {
            "precision_at_k": None,
            "recall_at_k": None,
            "ndcg_at_k": None,
            "mrr_at_k": None,
            "has_ground_truth": False,
        }

    top_predicted = predicted_ids[:k]
    relevant_ids = {product_id for product_id, rel in qrels_for_query.items() if rel > 0}
    if not relevant_ids:
        return {
            "precision_at_k": 0.0,
            "recall_at_k": None,
            "ndcg_at_k": None,
            "mrr_at_k": None,
            "has_ground_truth": False,
        }

    hits = sum(1 for product_id in top_predicted if product_id in relevant_ids)
    precision = hits / max(k, 1)
    recall = hits / max(len(relevant_ids), 1)

    predicted_rels = [qrels_for_query.get(product_id, 0) for product_id in top_predicted]
    ideal_rels = sorted((rel for rel in qrels_for_query.values() if rel > 0), reverse=True)[:k]
    dcg_value = _dcg(predicted_rels)
    idcg_value = _dcg(ideal_rels)
    ndcg = (dcg_value / idcg_value) if idcg_value > 0 else None

    mrr = 0.0
    for idx, product_id in enumerate(top_predicted, start=1):
        if qrels_for_query.get(product_id, 0) > 0:
            mrr = 1.0 / idx
            break

    return {
        "precision_at_k": round(precision, 6),
        "recall_at_k": round(recall, 6),
        "ndcg_at_k": round(ndcg, 6) if ndcg is not None else None,
        "mrr_at_k": round(mrr, 6),
        "has_ground_truth": True,
    }


def _aggregate_mode_metrics(rows: List[Dict[str, object]]) -> Dict[str, Optional[float]]:
    metric_names = ("precision_at_k", "recall_at_k", "ndcg_at_k", "mrr_at_k")
    aggregated: Dict[str, Optional[float]] = {}
    for metric_name in metric_names:
        values: List[float] = []
        for row in rows:
            metrics = row.get("metrics")
            if not isinstance(metrics, dict):
                continue
            value = metrics.get(metric_name)
            if isinstance(value, (float, int)):
                values.append(float(value))
        aggregated[metric_name] = round(sum(values) / len(values), 6) if values else None
    gt_rows = sum(
        1
        for row in rows
        if isinstance(row.get("metrics"), dict) and bool(row["metrics"].get("has_ground_truth"))
    )
    aggregated["ground_truth_coverage"] = round(gt_rows / max(len(rows), 1), 6)
    return aggregated


def _settings_for_mode(base: Settings, mode: str) -> Settings:
    if mode == "baseline_lexical":
        return replace(
            base,
            lexical_weight=1.0,
            semantic_weight=0.0,
            enable_llm_rerank=False,
        )
    if mode == "hybrid_semantic":
        return replace(
            base,
            lexical_weight=base.lexical_weight if base.lexical_weight > 0 else 0.35,
            semantic_weight=base.semantic_weight if base.semantic_weight > 0 else 0.65,
            enable_llm_rerank=False,
        )
    if mode == "hybrid_semantic_llm_rerank":
        return replace(
            base,
            lexical_weight=base.lexical_weight if base.lexical_weight > 0 else 0.35,
            semantic_weight=base.semantic_weight if base.semantic_weight > 0 else 0.65,
            enable_llm_rerank=True,
        )
    raise ValueError(f"Unknown evaluation mode: {mode}")


def run_evaluation(
    queries: Iterable[str] | None = None,
    service_factory: Callable[[Settings], RecommendationService] = RecommendationService,
) -> Dict[str, object]:
    base_settings = Settings.from_env()
    benchmark_queries = _load_benchmark_queries()
    if queries is not None:
        benchmark_queries = [
            BenchmarkQuery(query_id=f"CUSTOM_{idx + 1}", query=text, intent="general_search", k=5)
            for idx, text in enumerate(queries)
            if isinstance(text, str) and text.strip()
        ]
    qrels = _load_qrels()

    mode_outputs: Dict[str, Dict[str, object]] = {}
    for mode in RUN_MODES:
        mode_settings = _settings_for_mode(base_settings, mode)
        service = service_factory(mode_settings)
        rows: List[Dict[str, object]] = []

        for item in benchmark_queries:
            error_message: Optional[str] = None
            try:
                result = service.recommend(query=item.query, top_k=max(5, item.k)).to_dict()
            except (GenAIError, RuntimeError) as exc:
                result = {
                    "shortlist": [],
                    "best_value": None,
                    "explanation_mode": "deterministic",
                    "debug": {},
                }
                error_message = str(exc)

            shortlist = result.get("shortlist") if isinstance(result, dict) else []
            if not isinstance(shortlist, list):
                shortlist = []
            predicted_ids = [str(entry.get("product_id")) for entry in shortlist if isinstance(entry, dict)]
            predicted_ids = [product_id for product_id in predicted_ids if product_id]
            metrics = _metrics_at_k(predicted_ids=predicted_ids, qrels_for_query=qrels.get(item.query_id), k=item.k)

            best = result.get("best_value") if isinstance(result, dict) else None
            rows.append(
                {
                    "query_id": item.query_id,
                    "query": item.query,
                    "intent": item.intent,
                    "k": item.k,
                    "predicted_product_ids": predicted_ids[: item.k],
                    "best_title": best.get("title") if isinstance(best, dict) else None,
                    "best_product_id": best.get("product_id") if isinstance(best, dict) else None,
                    "explanation_mode": result.get("explanation_mode") if isinstance(result, dict) else None,
                    "rerank_used": bool(result.get("debug", {}).get("rerank_used")) if isinstance(result, dict) else False,
                    "error": error_message,
                    "metrics": metrics,
                }
            )

        mode_outputs[mode] = {
            "aggregate_metrics": _aggregate_mode_metrics(rows),
            "queries": rows,
        }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_modes": list(RUN_MODES),
        "query_count": len(benchmark_queries),
        "modes": mode_outputs,
    }


def save_evaluation_outputs(summary: Dict[str, object]) -> None:
    eval_dir = _root_dir() / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    summary_path = eval_dir / "metrics_summary.json"
    report_path = eval_dir / "metrics_report.txt"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: List[str] = []
    lines.append("Retrieval + Ranking Benchmark Summary")
    lines.append("=" * 60)
    lines.append(f"Generated at (UTC): {summary.get('generated_at_utc')}")
    lines.append(f"Run modes: {', '.join(summary.get('run_modes', []))}")
    lines.append("")

    modes = summary.get("modes", {})
    if isinstance(modes, dict):
        for mode in RUN_MODES:
            mode_payload = modes.get(mode)
            if not isinstance(mode_payload, dict):
                continue
            aggregate = mode_payload.get("aggregate_metrics", {})
            lines.append(f"[{mode}]")
            if isinstance(aggregate, dict):
                lines.append(f"precision@k: {aggregate.get('precision_at_k')}")
                lines.append(f"recall@k: {aggregate.get('recall_at_k')}")
                lines.append(f"ndcg@k: {aggregate.get('ndcg_at_k')}")
                lines.append(f"mrr@k: {aggregate.get('mrr_at_k')}")
                lines.append(f"ground_truth_coverage: {aggregate.get('ground_truth_coverage')}")
            lines.append("")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        output = run_evaluation()
        save_evaluation_outputs(output)
        print("Saved evaluation benchmark outputs to evaluation/metrics_summary.json and metrics_report.txt")
    except (GenAIError, RuntimeError, ValueError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
