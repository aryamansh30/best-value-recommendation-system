from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, List

from .config import Settings
from .genai import GenAIError
from .pipeline import RecommendationService

DEFAULT_QUERIES = [
    "Best electronics under $150",
    "Cheapest jewelry under $20",
    "Best t-shirt under $50",
    "Best laptop under $150",
]


def _load_queries_from_file() -> List[str]:
    root = Path(__file__).resolve().parents[2]
    query_file = root / "evaluation" / "queries.json"
    if not query_file.exists():
        return DEFAULT_QUERIES

    try:
        parsed = json.loads(query_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return DEFAULT_QUERIES

    if isinstance(parsed, list):
        sanitized = [item for item in parsed if isinstance(item, str) and item.strip()]
        if sanitized:
            return sanitized
    return DEFAULT_QUERIES


def run_evaluation(queries: Iterable[str] | None = None) -> List[dict]:
    settings = Settings.from_env()
    service = RecommendationService(settings)

    query_list = list(queries or _load_queries_from_file())
    outputs = []

    for query in query_list:
        result = service.recommend(query=query, top_k=5).to_dict()
        best = result.get("best_value")
        outputs.append(
            {
                "query": query,
                "best_title": best["title"] if best else None,
                "best_price": best["price"] if best else None,
                "best_score": best["score"] if best else None,
                "candidates_considered": result["candidates_considered"],
                "explanation": result["explanation"],
            }
        )

    return outputs


def save_evaluation_outputs(records: List[dict]) -> None:
    root = Path(__file__).resolve().parents[2]
    eval_dir = root / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    json_path = eval_dir / "sample_results.json"
    txt_path = eval_dir / "sample_qa.txt"

    json_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    lines = []
    for idx, item in enumerate(records, start=1):
        lines.append(f"Q{idx}: {item['query']}")
        lines.append(f"Best: {item['best_title']}")
        lines.append(f"Price: {item['best_price']}")
        lines.append(f"Score: {item['best_score']}")
        lines.append(f"Candidates considered: {item['candidates_considered']}")
        lines.append(f"Explanation: {item['explanation']}")
        lines.append("-" * 60)

    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    try:
        results = run_evaluation()
        save_evaluation_outputs(results)
        print(f"Saved {len(results)} evaluation results to evaluation/sample_results.json and sample_qa.txt")
    except (GenAIError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)
