# Baseline vs Improved (Short Comparison)

This section provides the requested baseline-vs-improved comparison in a reproducible way.

- Baseline: rank filtered candidates by lowest price only.
- Improved (this project): deterministic weighted best-value score from `src/app/ranking.py` using price efficiency, rating, relevance, review confidence, and discount.
- Dataset for this comparison: fixed fixture products embedded in `tests/test_ranking.py` (used to avoid live API drift).

Run context date: March 22, 2026.

| Query | Baseline (price-only) top result | Improved (best-value) top result | What this shows |
|---|---|---|---|
| Best wireless headphones under $150 | Budget Wired Earbuds ($24.99, rating 3.8, 120 reviews) | Wireless Over-Ear Headphones ($99.99, rating 4.4, 240 reviews, score 0.700) | Improved ranking avoids over-optimizing for lowest price and better captures value/quality tradeoff. |
| Cheapest protein snack bars | Protein Snack Bars 12 Pack ($18.50, rating 4.2) | Protein Snack Bars 12 Pack (score 0.500) | For explicit cheapest intent, improved logic remains aligned with baseline behavior. |
| Best-value keyboard for beginners | Mechanical Gaming Keyboard ($79.00, rating 4.6) | Mechanical Gaming Keyboard (score 0.500) | When one valid candidate exists, both methods are consistent; improved model is stable, not noisy. |

## Reproducibility Notes

- Comparison uses parsed query intent/category and the same deterministic constraint filtering as the main pipeline.
- The table is intentionally short because the assignment asks for simple but meaningful evaluation.
