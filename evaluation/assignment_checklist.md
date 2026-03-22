# Assignment Checklist (ENOVIA AI Intern)

This checklist maps each requested deliverable/criterion from the assignment PDF to concrete code, commands, and artifacts in this repository.

## A. Functional Requirements

- Accept natural-language shopping queries
  - Implemented in: `src/app/query_parser.py`
  - Verified by: `tests/test_query_parser.py`

- Retrieve from at least one free/public product source
  - Implemented in: `src/app/retrieval.py` (FakeStore API)
  - Verified by: `tests/test_retrieval.py`, `tests/test_pipeline.py` (live)

- Handle noisy/partial matches
  - Implemented in: retrieval scoring + category/term matching in `src/app/retrieval.py`
  - Additional no-match recovery: GenAI synonym fallback in `src/app/pipeline.py`
  - Verified by: `tests/test_pipeline_fallback.py`

- Filter and normalize results
  - Implemented in: `src/app/pipeline.py` (`_apply_constraints`) and `src/app/normalization.py`

- Rank by best value
  - Implemented in: `src/app/ranking.py`
  - Factors: price efficiency, rating, relevance, review confidence, discount
  - Verified by: `tests/test_ranking.py`

- Return shortlist + one recommendation + explanation
  - Implemented in: `src/app/pipeline.py` and `src/app/explainer.py`

## B. GenAI / RAG Usage (Optional)

- Where GenAI is used
  - Synonym expansion fallback only after deterministic no-match (`src/app/genai.py`, `src/app/pipeline.py`)

- Where GenAI is intentionally not used
  - Filtering constraints
  - Price comparison
  - Ranking formula/math

- Validation
  - `tests/test_pipeline_fallback.py` confirms deterministic-first behavior.

## C. Evaluation

- Evaluation runner
  - `PYTHONPATH=src python -m app.evaluation`
  - Writes:
    - `evaluation/sample_results.json`
    - `evaluation/sample_qa.txt`
  - Additional short comparison doc:
    - `evaluation/baseline_vs_improved.md`

- Included sample queries (4 total, includes required >=3)
  - `Best electronics under $150`
  - `Cheapest jewelry under $20`
  - `Best t-shirt under $50`
  - `Best laptop under $150` (no-match guardrail)

## D. Required Deliverables

- Runnable code
  - CLI: `PYTHONPATH=src python -m app.recommend --query 'Best electronics under $150'`
  - API: `PYTHONPATH=src uvicorn app.api:app --reload`

- README
  - Main document: `README.md`

- Setup instructions
  - Included in `README.md` section "Setup"
  - Also supported via `make setup`

- Results for at least 3 test queries
  - Included in `evaluation/sample_results.json` and `evaluation/sample_qa.txt`

- Architecture explanation
  - Included in `README.md` sections "Project Structure" and "Design and Reasoning"

- Evaluation results
  - Included in `evaluation/sample_results.json`, `evaluation/sample_qa.txt`, and `evaluation/baseline_vs_improved.md`

## E. Evaluation Criteria Alignment

- Retrieval quality (relevant products, handles noise)
  - Retrieval + category-aware matching + dedupe in `src/app/retrieval.py`

- Best-value logic (sensible ranking, respects constraints)
  - Constraints in `src/app/pipeline.py`
  - Ranking formula in `src/app/ranking.py`

- Engineering clarity (easy to run, clean code)
  - Modular pipeline with clear separation of concerns
  - Make targets: `make test`, `make eval`, `make recommend`, `make api`

- Communication (clear explanations, honest tradeoffs)
  - Explanation output in `src/app/explainer.py`
  - Tradeoffs/limitations documented in `README.md`
