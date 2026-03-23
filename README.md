# Product Retrieval & Best-Value Recommendation

This repository implements a recommendation system that:
- accepts natural-language shopping queries,
- retrieves products from a local CSV catalog (`data/enhanced_fakestore_products.csv`),
- handles noisy/partial matches,
- performs hybrid lexical and semantic retrieval with embedding fallback guardrails,
- filters and normalizes candidates,
- ranks by a transparent best-value formula,
- optionally applies LLM re-ranking in A/B mode,
- returns a shortlist, one best recommendation, and a grounded explanation with evidence citations.


## 1. Requirement Coverage

| Required item | Implemented in this repo |
|---|---|
| Natural-language input queries | `src/app/query_parser.py` |
| One source dataset | CSV catalog loader in `src/app/retrieval.py` |
| Retrieve relevant products and handle noisy matches | Hybrid lexical and semantic retrieval, weighted fusion, and fallback guardrails in `src/app/retrieval.py` |
| Filter and normalize results | `src/app/pipeline.py` + `src/app/normalization.py` |
| Best-value ranking | Deterministic weighted scoring in `src/app/ranking.py` |
| Return shortlist, single recommendation and explanation | `src/app/pipeline.py` + `src/app/explainer.py` |
| Retrieval-augmented grounded explanation (RAG style) | `src/app/pipeline.py` + `src/app/genai.py` |
| LLM rerank A/B mode | Optional rerank in `src/app/pipeline.py` + `src/app/genai.py` |
| Evaluation | `src/app/evaluation.py` + `evaluation/benchmark_queries.jsonl` + `evaluation/qrels.jsonl` |


## 2. Project Structure

- `src/app/query_parser.py`: deterministic parsing for category, budget, intent, and filters.
- `src/app/retrieval.py`: product retrieval from local CSV catalog + optional RapidAPI extension
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
- Internet access only if using optional embedding/LLM providers or RapidAPI extension

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

## 5. Design and Reasoning

### 5.1 Retrieval strategy
- Primary source: local CSV catalog (`data/enhanced_fakestore_products.csv`).
- Category filtering is applied when query category maps cleanly.
- Intent-aware price sorting (`cheapest` asc, `premium/best_value` desc) is applied before lexical/semantic scoring.
- Retrieval scoring uses weighted fusion:
  - lexical relevance from overlap/title/category matching,
  - semantic relevance from embedding cosine similarity over title/category/description.
- Fusion defaults: `lexical_weight=0.35`, `semantic_weight=0.65` (auto-normalized).
- If embeddings are unavailable/invalid, retrieval falls back to lexical-only deterministically.
- Embeddings are cached on disk (`EMBEDDING_CACHE_PATH`) keyed by provider/model/product/text hash.
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

### 5.3 GenAI 
GenAI is used in two bounded places:
- synonym expansion to retry retrieval (`src/app/genai.py`, called from `src/app/pipeline.py`).
- optional LLM reranking of top-N deterministic candidates (A/B mode with strict JSON validation and fallback).
- grounded explanation generation from retrieved evidence after ranking.

If reranking or grounded generation fails/returns invalid payloads, the pipeline falls back to deterministic outputs.

### 5.4 RAG evidence + citation contract
After ranking, the pipeline builds an evidence pack from top ranked products (`E1..E5`) and asks the model to explain using only this context.

Model output contract:
```json
{
  "explanation": "Silicon Power SSD is the strongest value at this budget [E1], with WD Gaming Drive as a close alternative [E2].",
  "citations": [
    {"evidence_id": "E1", "product_id": "9", "field": "price", "quote_or_value": "109.0"},
    {"evidence_id": "E2", "product_id": "10", "field": "rating", "quote_or_value": "4.8"}
  ],
  "used_evidence_ids": ["E1", "E2"],
  "limitations": "Based only on products retrieved in this run."
}
```

Response payloads include:
- `explanation_mode`: `deterministic` or `rag_grounded`
- `citations`: validated grounding citations
- `grounding`: `grounded_generation_used`, `evidence_count`, `citation_count`, `fallback_reason`


## 6. Evaluation

Run evaluation:
```bash
PYTHONPATH=src python -m app.evaluation
```

Artifacts written:
- `evaluation/metrics_summary.json`
- `evaluation/metrics_report.txt`
- input benchmark assets:
  - `evaluation/benchmark_queries.jsonl`
  - `evaluation/qrels.jsonl`

Run modes:
1. `baseline_lexical`
2. `hybrid_semantic`
3. `hybrid_semantic_llm_rerank`

Metrics tracked:
- `precision@k`
- `recall@k`
- `ndcg@k`
- `mrr@k`
- ground-truth coverage

## 7. Tests and Validation Walkthrough

Run all tests:
```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

### Test coverage by file
- `tests/test_query_parser.py`
  - validates budget extraction, intent detection, category mapping, and t-shirt parsing.
- `tests/test_retrieval.py`
  - validates CSV-backed category mapping and intent-based sorting.
- `tests/test_ranking.py`
  - validates cheapest-intent behavior and score breakdown presence.
- `tests/test_genai.py`
  - validates robust extraction of explanation text, grounded payload validation, and rerank payload validation.
- `tests/test_pipeline_fallback.py`
  - validates deterministic-first behavior and GenAI synonym fallback activation only after no-match.
- `tests/test_embeddings.py`
  - validates cosine similarity, weight normalization, and embedding cache behavior.
- `tests/test_retrieval_semantic.py`
  - validates hybrid retrieval scoring and lexical fallback behavior.
- `tests/test_pipeline_rerank.py`
  - validates LLM rerank A/B behavior and fallback on invalid rerank payloads.
- `tests/test_evaluation_metrics.py`
  - validates precision/recall/NDCG/MRR calculation and multi-mode evaluation outputs.
- `tests/test_pipeline.py`
  - integration tests against the current local catalog; skipped automatically if optional live dependencies are unavailable.

### Why this test strategy
- Core logic is validated with deterministic unit tests.
- External dependency behavior is covered via explicit live integration tests without making local runs brittle.

## 8. Environment Configuration

`.env` controls optional provider/source integrations.

Key variables:
- `CATALOG_CSV_PATH`, `REQUEST_TIMEOUT_SEC`, `SSL_CA_BUNDLE`
- `LLM_PROVIDER` (`ollama` or `openai`)
- `OLLAMA_*` / `OPENAI_*`
- `EMBEDDING_PROVIDER` (`local`, `openai`, or `ollama`)
- `OPENAI_EMBEDDING_MODEL`, `OLLAMA_EMBEDDING_MODEL`, `LOCAL_EMBEDDING_MODEL`
- `EMBEDDING_LOCAL_FALLBACK`, `EMBEDDING_CACHE_PATH`
- `SEMANTIC_WEIGHT`, `LEXICAL_WEIGHT`
- `ENABLE_LLM_RERANK`, `LLM_RERANK_TOP_N`, `LLM_RERANK_WEIGHT`
- `USE_RAPIDAPI` and `RAPIDAPI_*` (optional extension)

## 9. Limitations and Improvements

Current limitations:
- The current catalog is a toy-level synthetic dataset, so qrels cover only a narrow catalog.
- Embedding and LLM calls depend on configured provider availability and network access.

Potential next improvements:
1. Expand qrels coverage and add inter-annotator review for stronger benchmark trust.
2. Add broader product sources and source-confidence fusion beyond the single CSV catalog.

## 10. Run Commands

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
