# ENOVIA Take-Home Assignment — Consolidated Implementation Plan

## 1. Purpose

Build a small, clear, runnable **Product Retrieval & Best-Value Recommendation System** that accepts a natural-language shopping query, retrieves products from at least one free/public source, handles noisy matches, filters and normalizes results, ranks them using a transparent **best-value** formula, and returns a shortlist plus one recommended product with a short explanation.

This plan is intentionally written in **plan mode**: it focuses on the exact work, sequence, design decisions, and deliverables needed to produce a strong submission within the expected time window.

---

## 2. What the assignment is really evaluating

The assignment is not mainly about building a flashy product. It is evaluating whether the solution shows:

- structured query handling
- clear separation between retrieval and ranking
- sensible “best value” logic
- handling of noisy/partial product matches
- thoughtful and limited use of GenAI/RAG, only where it helps
- clean engineering structure
- honest evaluation and tradeoff discussion

So the solution should optimize for **clarity, determinism, explainability, and ease of execution** over UI polish or unnecessary complexity.

---

## 3. Recommended submission strategy

## Recommended build choice
Use a **CLI-based Python project** with a **public dataset** as the primary product source.

### Why this is the best choice
- More reliable than depending entirely on external APIs during evaluation
- Easier to make reproducible
- Easier to document and test against at least 3 queries
- Fits the assignment note that a CLI script, notebook, or small API is sufficient
- Lets the solution emphasize retrieval, normalization, ranking, and reasoning

## Recommended baseline architecture
1. **Query parser**
2. **Retriever**
3. **Filtering + normalization**
4. **Ranker**
5. **Explainer**
6. **Main entrypoint / CLI**
7. **README + evaluation**

## Recommended GenAI boundary
Use GenAI only as an **optional enhancement** for:
- query parsing of ambiguous input
- synonym expansion
- final explanation phrasing

Do **not** use GenAI for:
- filtering decisions
- score computation
- final business logic
- constraint satisfaction

This keeps the strongest part of the system deterministic and easy to defend.

---

## 4. Scope and success criteria

A successful final submission must do all of the following:

### Required functionality
- Accept a natural-language query such as:
  - “Best headphones under $150”
  - “Cheapest protein snack bars”
  - “Best-value keyboard for beginners”
- Use at least one free/public product API or dataset
- Retrieve relevant products
- Handle noisy and partial matches
- Filter products based on constraints like budget and category
- Normalize product attributes into a consistent schema
- Rank products using a transparent best-value formula
- Return:
  - a shortlist of relevant products
  - one best-value recommendation
  - a short explanation

### Required deliverables
- runnable code
- README.md
- setup instructions
- results for at least 3 test queries
- architecture explanation
- evaluation results

### README must explicitly explain
- data source(s) used and why
- how retrieval works
- how “best value” is defined
- where GenAI was used, if anywhere
- what remained deterministic
- how the system was evaluated
- limitations and improvements

---

## 5. Final design decision

## Preferred final version
A **deterministic baseline with optional GenAI hooks**.

That means the final submitted version should definitely work without GenAI.  
Then, if time permits, add a lightweight optional GenAI layer for query parsing and/or explanation generation.

### Why this is the safest plan
- Best aligns with “clarity and reasoning over polish”
- Avoids dependence on LLM availability
- Makes the ranking logic defendable in an interview or review
- Demonstrates stronger engineering judgment than overusing AI

---

## 6. End-to-end workflow

## Workflow overview

```text
User Query
   ↓
Query Understanding
   ↓
Query Expansion / Category Mapping
   ↓
Product Retrieval from Public Source
   ↓
Candidate Cleaning + Deduplication
   ↓
Filtering by Constraints
   ↓
Normalization to Canonical Schema
   ↓
Relevance Scoring
   ↓
Best-Value Ranking
   ↓
Top-5 Shortlist + Best Recommendation
   ↓
Explanation Generation
```

## Functional workflow in plain English

1. User enters a shopping query.
2. The system extracts structured intent:
   - category
   - budget
   - intent type
   - filters
3. The system converts that into retrieval terms.
4. It fetches candidate products from the chosen dataset or API.
5. It removes duplicates, irrelevant accessories, invalid prices, and bad records.
6. It standardizes the data into one clean product schema.
7. It computes relevance and other ranking features.
8. It applies a deterministic best-value score.
9. It returns the top shortlist and the final recommendation.
10. It explains why the winning item ranked highest.

---

## 7. Detailed module plan

## 7.1 Query Understanding Layer

### Goal
Convert natural language into structured intent.

### Input example
`Best wireless headphones under $150`

### Output schema
```json
{
  "category": "headphones",
  "budget": 150,
  "intent": "best_value",
  "filters": {
    "type": "wireless"
  }
}
```

### Planned implementation
#### Baseline deterministic parser
- lowercase normalization
- punctuation cleanup
- regex-based budget extraction
- keyword mapping for category
- keyword mapping for intent
- optional filter extraction for terms like:
  - wireless
  - beginner
  - gaming
  - premium
  - cheap / cheapest

#### Optional GenAI enhancement
Use a schema-constrained LLM prompt only for:
- ambiguous category mapping
- synonym resolution
- query normalization

### Fallback logic
If GenAI parsing fails or returns invalid JSON:
- fall back to regex + keyword parser

### Output fields
- `category`
- `budget`
- `intent`
- `filters`
- optional `confidence`

### Key rule
The parser can use AI. The final decision logic cannot depend on AI.

---

## 7.2 Product Retrieval Layer

### Goal
Fetch relevant candidates from one public source.

### Recommended source choice
Choose one:
1. **Public dataset (recommended)** for reproducibility
2. Public API as an optional extension

### Retrieval activities
- load product data
- search by category and expanded keywords
- retrieve top 20–50 candidate rows
- deduplicate results
- retain enough breadth for later ranking

### Retrieval logic
Use a hybrid deterministic retrieval score based on:
- keyword overlap
- partial title match
- fuzzy matching
- optional category boost
- optional attribute boost

### Example scoring idea
```text
retrieval_score =
  0.5 * keyword_match +
  0.3 * fuzzy_match +
  0.2 * attribute/category match
```

### Noise-handling rules
Explicitly exclude likely irrelevant accessory terms when needed, such as:
- stand
- holder
- case
- cover

This helps prevent false matches like “headphone stand” when the user wants headphones.

### Optional GenAI usage
Only for semantic query expansion, such as:
- headphones → earbuds, bluetooth headphones
- protein bars → protein snack bars

Do not let GenAI perform the actual ranking.

---

## 7.3 Filtering and Normalization Layer

### Goal
Turn messy retrieved records into a clean, comparable product set.

### Canonical product schema
```json
{
  "product_id": "string",
  "title": "string",
  "category": "string",
  "price": "float",
  "rating": "float",
  "review_count": "int",
  "discount": "float",
  "retrieval_score": "float"
}
```

### Cleaning activities
- parse string prices into floats
- remove rows with invalid or missing core identifiers
- normalize category text
- standardize title formatting
- fill missing rating with neutral default, e.g. 3.5
- fill missing discount with 0
- standardize review counts
- optionally convert currencies if needed
- remove duplicates by product id or normalized title

### Constraint filters
- price must be within budget when a budget exists
- product category should match intended category
- optional filter match should be respected when reasonable

### Important rule
Filtering happens **before** final ranking.

---

## 7.4 Ranking Engine

### Goal
Select the best-value product using transparent scoring.

### Ranking philosophy
“Best value” is not the cheapest product.  
It is the best balance of:
- price
- rating
- discount
- relevance
- quantity/size if available

### Recommended normalized features
- `normalized_rating`
- `normalized_discount`
- `normalized_relevance`
- `normalized_price`
- optional `normalized_quantity_value`

### Baseline scoring formula
```text
best_value_score =
    0.40 * normalized_rating +
    0.20 * normalized_discount +
    0.20 * normalized_relevance -
    0.20 * normalized_price
```

### Notes
- higher rating is better
- higher discount is better
- higher relevance is better
- lower price is better, so price acts as a penalty
- if quantity/size exists for a product category, incorporate it carefully rather than forcing it globally

### Normalization approach
Use min-max scaling over the candidate set:
```text
(x - min) / (max - min)
```

### Safeguards
- handle zero-variance columns safely
- avoid division-by-zero
- document assumptions in README

### Output
- Top 5 shortlist
- 1 best-value product
- score breakdown for explainability

---

## 7.5 Explanation Layer

### Goal
Generate a short explanation of why the chosen product won.

### Recommended baseline
Use a template-based explanation from computed features.

### Example format
“This product was selected because it had one of the strongest overall balances of rating, relevance, and discount while staying within budget. It was not necessarily the cheapest option, but it produced the highest best-value score among the valid candidates.”

### Optional GenAI enhancement
Feed the structured score breakdown into an LLM and ask it to rewrite the explanation more naturally.

### Key rule
The explanation may be AI-written, but the underlying reasoning must come from deterministic scores.

---

## 7.6 Interface Layer

### Preferred interface
A simple CLI.

### Example usage
```bash
python main.py "Best wireless headphones under $150"
```

### CLI output should include
- parsed query
- shortlist table
- selected best-value product
- short explanation

### Optional extension
A tiny REST endpoint such as `POST /recommend` can be added only if time remains.

---

## 8. Implementation activities by phase

## Phase 1 — Project setup
### Activities
- create project folder structure
- choose public dataset or API
- install dependencies
- create requirements file
- create baseline README skeleton

### Output
Working repo scaffold.

---

## Phase 2 — Data ingestion and schema definition
### Activities
- load dataset / API response
- inspect available fields
- define canonical schema
- document all field assumptions
- create a small sample dataset for quick testing if needed

### Output
Stable input layer and normalized schema target.

---

## Phase 3 — Query parser
### Activities
- implement regex budget extraction
- build intent keyword map
- build category keyword map
- extract filters
- add deterministic fallback logic
- optionally add GenAI parser wrapper

### Output
Structured query object.

---

## Phase 4 — Retrieval pipeline
### Activities
- convert parsed query into retrieval terms
- run keyword and fuzzy matching over titles/categories
- deduplicate candidates
- cap results at 20–50
- add exclusion rules for common irrelevant accessory terms

### Output
Candidate product set.

---

## Phase 5 — Filtering and normalization
### Activities
- clean prices
- fill missing values
- filter by budget
- filter by category
- standardize ratings, discounts, review counts
- normalize records into canonical schema

### Output
Comparable product table ready for ranking.

---

## Phase 6 — Ranking engine
### Activities
- compute retrieval/relevance score
- normalize ranking features
- apply best-value formula
- sort products
- capture score breakdown per product

### Output
Ranked shortlist + winning product.

---

## Phase 7 — Explanation generation
### Activities
- create deterministic explanation template
- optionally add GenAI rewrite mode
- make sure explanation references actual score factors

### Output
Human-readable justification.

---

## Phase 8 — Evaluation
### Activities
- test at least 3 required queries
- compare against a baseline such as price-only sorting
- record whether results are relevant
- verify budget constraints are respected
- note failure cases and tradeoffs

### Suggested evaluation queries
- Best headphones under $150
- Cheapest protein bars
- Best keyboard for beginners

### Metrics
- Retrieval quality: relevant products in top K
- Noise handling: irrelevant products removed or reduced
- Constraint satisfaction: budget respected
- Ranking correctness: result makes intuitive sense
- Baseline comparison: price-only vs multi-factor scoring

### Output
Evaluation table for README.

---

## Phase 9 — Packaging and documentation
### Activities
- write setup instructions
- document architecture
- describe retrieval logic
- explain best-value formula
- explain GenAI boundaries
- summarize evaluation
- list limitations and future improvements

### Output
Submission-ready repo.

---

## 9. Recommended project structure

```text
project/
│
├── main.py
├── parser.py
├── retriever.py
├── normalizer.py
├── ranker.py
├── explainer.py
├── utils.py
│
├── data/
├── tests/
│
├── README.md
└── requirements.txt
```

---

## 10. Evaluation plan

## Baseline vs improved comparison

### Baseline
Sort valid products by lowest price only.

### Improved system
Use the best-value scoring formula across rating, discount, relevance, and price.

### Why this comparison matters
It clearly demonstrates that:
- cheapest is not always best value
- ranking logic is intentional
- the improved system is meaningfully better than naive search + price sort

## Suggested evaluation table
| Query | Best Product | Price Constraint Met? | Relevant? | Better than Price-Only? | Notes |
|---|---|---:|---:|---:|---|
| Best headphones under $150 | TBD | Yes | Yes | Yes | Strong rating/relevance balance |
| Cheapest protein bars | TBD | N/A | Yes | Moderate | May look similar to baseline |
| Best keyboard for beginners | TBD | N/A | Yes | Yes | Relevance matters strongly |

---

## 11. Limitations to acknowledge

The submission should explicitly state realistic limitations, such as:
- limited product coverage from a single source
- ratings may be missing or unreliable
- discounts may be incomplete or synthetic
- category mapping may be imperfect
- quantity/size may not exist for all categories
- no personalization
- no real-time price updates if using a static dataset
- optional GenAI output may vary slightly if enabled

Being honest here is a strength, not a weakness.

---

## 12. Nice-to-have enhancements if time remains

Only after the full baseline works:

1. GenAI parser for ambiguous queries  
2. GenAI explanation rewrite  
3. Embedding-based retrieval for semantic matching  
4. Category-aware taxonomy mapping  
5. Quantity/size-aware scoring for grocery items  
6. REST API wrapper  
7. Basic unit tests for parser, normalizer, and ranker

These are enhancements, not the core deliverable.

---

## 13. Time-boxed execution plan

## 6–10 hour implementation breakdown

### Hour 1
- finalize source choice
- create project scaffold
- inspect dataset/API schema

### Hours 2–3
- implement parser
- implement basic retrieval
- test candidate generation

### Hours 4–5
- implement filtering + normalization
- implement ranking formula
- debug edge cases

### Hour 6
- implement explanation layer
- wire end-to-end CLI flow

### Hours 7–8
- run 3+ evaluation queries
- compare with baseline
- collect outputs for README

### Hours 9–10
- polish README
- clean code
- document limitations and optional GenAI usage

---

## 14. Final submission checklist

## Code
- [ ] runnable project
- [ ] clear module separation
- [ ] deterministic ranking
- [ ] optional GenAI clearly isolated

## Functionality
- [ ] natural-language query input
- [ ] public source retrieval
- [ ] noisy/partial match handling
- [ ] filtering and normalization
- [ ] shortlist + best recommendation
- [ ] explanation

## Documentation
- [ ] setup instructions
- [ ] source choice explained
- [ ] retrieval logic explained
- [ ] best-value formula explained
- [ ] GenAI usage explained
- [ ] deterministic parts identified
- [ ] evaluation results included
- [ ] limitations included

## Evaluation
- [ ] at least 3 queries tested
- [ ] baseline vs improved comparison shown
- [ ] constraints verified
- [ ] honest tradeoffs documented

---

## 15. Final recommendation

The strongest version of this assignment is:

- **simple**
- **modular**
- **deterministic at its core**
- **clear about best-value logic**
- **careful with GenAI**
- **easy to run**
- **easy to explain**

So the working plan should be:

1. Build a clean deterministic baseline first.
2. Make retrieval, normalization, and ranking rock-solid.
3. Use GenAI only as a small, well-bounded enhancement.
4. Spend real effort on README, evaluation, and reasoning.
5. Prefer clarity and trustworthiness over extra features.

That is the version most likely to look thoughtful, engineering-driven, and aligned with the assignment expectations.
