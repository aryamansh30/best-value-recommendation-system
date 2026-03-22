from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .genai import GenAIError
from .pipeline import RecommendationService


def _print_human_readable(result: dict) -> None:
    parsed = result["parsed_query"]
    shortlist = result["shortlist"]

    print(f"Query: {parsed['query']}")
    print(
        f"Parsed -> category={parsed['category']}, budget={parsed['budget']}, intent={parsed['intent']}, "
        f"filters={parsed['filters']}"
    )
    print(f"Candidates considered: {result['candidates_considered']}")

    if not shortlist:
        print("No recommendations found after constraints.")
        print(f"Explanation: {result['explanation']}")
        return

    print("\nShortlist:")
    for idx, item in enumerate(shortlist, start=1):
        print(
            f"{idx}. {item['title']} | ${item['price']:.2f} | rating={item['rating']:.1f} "
            f"| score={item['score']:.3f} | source={item['source']}"
        )

    print("\nBest Value:")
    best = result["best_value"]
    print(
        f"{best['title']} | ${best['price']:.2f} | rating={best['rating']:.1f} | score={best['score']:.3f}"
    )
    print(f"\nExplanation: {result['explanation']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run best-value recommendation from CLI.")
    parser.add_argument("--query", required=True, help="Natural language shopping query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of recommendations to return")
    parser.add_argument("--use-rapidapi", action="store_true", help="Enable optional RapidAPI source")
    parser.add_argument("--json", action="store_true", help="Print raw JSON output")

    args = parser.parse_args()

    settings = Settings.from_env()
    service = RecommendationService(settings)
    try:
        result = service.recommend(
            query=args.query,
            top_k=args.top_k,
            use_rapidapi=args.use_rapidapi,
        ).to_dict()
    except (GenAIError, RuntimeError) as exc:
        print(f"Recommendation request failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        _print_human_readable(result)


if __name__ == "__main__":
    main()
