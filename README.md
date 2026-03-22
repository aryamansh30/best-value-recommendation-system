# Product Retrieval & Best-Value Recommendation

This repository implements a recommendation system that:
- accepts natural-language shopping queries,
- retrieves products from a free/public source (FakeStore API),
- handles noisy/partial matches,
- filters and normalizes candidates,
- ranks by a transparent best-value formula,
- returns a shortlist, one best recommendation, and an explanation.


## 1. Requirement Coverage

| Required item | Implemented in this repo |
|---|---|
| Natural-language input queries | `src/app/query_parser.py` |
| One free/public source | FakeStore API in `src/app/retrieval.py` |
| Retrieve relevant products and handle noisy matches | Retrieval scoring, dedupe and term matching in `src/app/retrieval.py` |
| Filter and normalize results | `src/app/pipeline.py` + `src/app/normalization.py` |
| Best-value ranking | Deterministic weighted scoring in `src/app/ranking.py` |
| Return shortlist, single recommendation and explanation | `src/app/pipeline.py` + `src/app/explainer.py` |
| Simple evaluation | `src/app/evaluation.py` writes `evaluation/sample_results.json` and `evaluation/sample_qa.txt` |
| README with architecture + tradeoffs | This file + `evaluation/assignment_checklist.md` |


## 2. Project Structure

- `src/app/query_parser.py`: deterministic parsing for category, budget, intent, and filters.
- `src/app/retrieval.py`: product retrieval from FakeStore 
- `src/app/normalization.py`: canonical product schema normalization.
- `src/app/ranking.py`: deterministic best-value scoring formula.
- `src/app/explainer.py`: explanation generation from ranking breakdown.
- `src/app/pipeline.py`: orchestration of parse then retrieve then normalize then filter then rank and then explain.
- `src/app/recommend.py`: CLI entrypoint.
- `src/app/api.py`: FastAPI entrypoint (`POST /recommend`).
- `src/app/evaluation.py`: evaluation runner for assignment queries.
- `tests/`: unit tests and live integration tests.

## 3. Setup

### Prerequisites
- Python 3.10+
- Internet access for FakeStore API integration/evaluation

### Install
```bash
cd best-value-recommendation-system
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```


## 4. Run the Application

### CLI
```bash
PYTHONPATH=src python -m app.recommend --query 'Best electronics under $150' --top-k 5
```

JSON output:
```bash
PYTHONPATH=src python -m app.recommend --query 'Best t-shirt under $50' --json
```


Example request:
```bash
curl -X POST http://127.0.0.1:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"query":"Cheapest jewelry under $20","top_k":5}'
```

## 5. Design and Reasoning

### 5.1 Retrieval strategy
- Primary source: FakeStore API (`https://fakestoreapi.com/products`).
- Category-specific routes are used when query category maps cleanly.
- Global route fallback is used when category routes fail or return no data.
- Retrieval scoring combines lexical overlap with category-aware matching to suppress unrelated products.
- Candidate deduplication is applied before ranking.

### 5.2 Best-value definition (deterministic)
Each candidate is normalized and scored using weighted components:
- price efficiency (lower price is better),
- rating,
- relevance (retrieval signal),
- review confidence,
- discount.

Intent-aware weighting:
- `cheapest`: strongly price-weighted.
- `best_value`: balanced default.
- `premium`: quality/reputation weighted.

This scoring is fully deterministic and explainable in `src/app/ranking.py`.

### 5.3 GenAI boundary 
GenAI is not used for ranking math, filtering, or price comparison.

GenAI is used only as a fallback when deterministic retrieval yields no valid match:
- synonym expansion to retry retrieval (`src/app/genai.py`, called from `src/app/pipeline.py`).


## 6. Evaluation

Run evaluation:
```bash
PYTHONPATH=src python -m app.evaluation
```

Artifacts written:
- `evaluation/sample_results.json`
- `evaluation/sample_qa.txt`
- `evaluation/baseline_vs_improved.md` (short baseline vs improved comparison table)

Current sample queries (`evaluation/queries.json`):
1. `Best electronics under $150`
2. `Cheapest jewelry under $20`
3. `Best t-shirt under $50`
4. `Best laptop under $150` (intentional no-match guardrail)

## 7. Tests and Validation Walkthrough

Run all tests:
```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### Test coverage by file
- `tests/test_query_parser.py`
  - validates budget extraction, intent detection, category mapping, and t-shirt parsing.
- `tests/test_retrieval.py`
  - validates FakeStore URL construction, category route mapping, and intent-based sorting.
- `tests/test_ranking.py`
  - validates cheapest-intent behavior and score breakdown presence.
- `tests/test_genai.py`
  - validates robust extraction of explanation text from model responses.
- `tests/test_pipeline_fallback.py`
  - validates deterministic-first behavior and GenAI synonym fallback activation only after no-match.
- `tests/test_pipeline.py`
  - live integration tests against FakeStore; skipped automatically if live dependencies are unavailable.

### Why this test strategy
- Core logic is validated with deterministic unit tests.
- External dependency behavior is covered via explicit live integration tests without making local runs brittle.

## 8. Environment Configuration

`.env` controls optional provider/source integrations.

Key variables:
- `FAKESTORE_URL`, `REQUEST_TIMEOUT_SEC`, `SSL_CA_BUNDLE`
- `LLM_PROVIDER` (`ollama` or `openai`)
- `OLLAMA_*` / `OPENAI_*`
- `USE_RAPIDAPI` and `RAPIDAPI_*` (optional extension)

## 9. Limitations and Improvements

Current limitations:
- FakeStore dataset is small and category-limited.
- Evaluation is simple (query-level outcomes), not full ranking metrics.

Potential next improvements:
1. Add richer retrieval features (fuzzy scoring).
2. Add broader product sources and confidence-weighted source fusion.

## 10. Useful Commands

With Makefile:
```bash
make setup
make test
make eval
make recommend QUERY='Best electronics under $150'
```

Direct Python commands:
```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python -m app.evaluation
PYTHONPATH=src python -m app.recommend --query 'Best electronics under $150'
```