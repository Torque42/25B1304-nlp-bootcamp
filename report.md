## Q1

BM25 did better on 1156 (DESMOND) and 673 (LDL causality): it ranked the gold abstract first, while MiniLM ranked it 40th and fifth. DESMOND is a distinctive name, and the LDL claim closely matches the abstract's wording. MiniLM did better on 1311 (lncRNA translation) and 55 (AMPK and lung fibrosis), ranking the gold abstract first versus BM25's 24th and eighth. These abstracts use more specific terms, including lincRNA-p21 and metformin/bleomycin. Semantic matching likely helped connect those terms to the claims. The supplied validation rankings show all four comparisons.

## Q2

Hybrid fusion gave the biggest gain: nDCG@10 rose from 0.8509 for MedCPT retrieval to 0.8961, an increase of 0.0452. Estimated latency rose from 59.9 to 81.0 ms per query. Reranking the top 20 added another 0.0148 nDCG, reaching 0.9109, but cost an extra 950.9 ms per query. Total estimated latency was 1032.0 ms. These timings use 30 warmed queries on MPS; totals add the separately measured stages. Fusion therefore offered the better gain for its latency cost, while reranking delivered the highest score among the eligible configurations tested.

## Q3

Yes. Hard negatives reached 0.8396 validation nDCG@10, compared with 0.8247 for random negatives and 0.8155 before fine-tuning. Both runs started from the same MiniLM weights and used the same one-epoch schedule. Their mean training losses were 0.2056 and 0.0047 respectively; the harder examples explain why a higher loss can accompany better retrieval. I checked training losses and validation scores, but one epoch gives too little evidence to diagnose overfitting. Both validation scores improved. Tracking training and validation performance across several checkpoints would give a stronger answer.

## Q4

Four of the 74 incorrect validation claims, or 5.41%, were retrieval misses: 531, 559, 738 and 1003. Their missing gold abstracts were in the top 100, but below the top-five cutoff used by the verifier. I count these as retrieval_miss because the verifier never saw the required evidence. If a claim also had classification errors, the missing evidence took precedence. The remaining counts were 11 entity mismatches, five numerical errors, three negation errors and 51 other errors. None clearly required combining multiple documents. Correct claims received no error category.

## Q5

I would train a sentence selector to pick the evidence most relevant to each claim before passing it to NLI. The current verifier reaches 0.5827 F1, yet 43 incorrect claims have gold documents that reached NLI and were rejected or mislabeled. A selector could keep useful sentences within the token limit and remove distracting text. Training evidence annotations provide supervision. I would keep retrieval, candidate depth and the calibration procedure fixed to test whether the selector actually helps. This remains a proposed experiment, not a measured improvement.
