# Final submitted results

These are the results for the submitted system on the supplied validation set. The same split was used to select settings and thresholds. Retrieval averages 106 claims with evidence; verification includes all 162 claims.

| Run | nDCG@10 | Recall@100 | MRR@100 |
|---|---:|---:|---:|
| bm25 | 0.836567 | 0.990566 | 0.815069 |
| general | 0.815549 | 0.990566 | 0.799307 |
| scientific | 0.850892 | 1.000000 | 0.812675 |
| hybrid | 0.896061 | 1.000000 | 0.871502 |
| rerank | 0.910855 | 1.000000 | 0.889640 |
| random | 0.824713 | 0.990566 | 0.807299 |
| hard | 0.839577 | 0.990566 | 0.823630 |

Verification: precision 0.569231; recall 0.596774; F1 0.582677; TP 74, FP 56, FN 50. Exact-set errors: 74/162. Candidate cutoff 5; evidence threshold 0.9; conditional polarity threshold 0.7.

Categories: other 51, entity_mismatch 11, numerical 5, negation 3, retrieval_miss 4, needs_multiple_docs 0. Retrieval misses: 4/74 = 5.41%, all within top 100 but below verifier cutoff.

For the final settings and outputs, see `results.json`, `report.md`, `predictions/`, and `nli_selection.json`. The final category decisions and review process are recorded in `final-category-decisions.json` and `final-review-provenance.json`. Other files document earlier experiments, including failures. Absolute paths in those logs refer to the original machine; they are not needed to run the code.
