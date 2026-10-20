# Experiment notes — 13 October 2026

These notes describe the experiment extension that produced the submitted configuration. The initial work window began at 12:44 UTC and was extended after an interruption. The experiment records are under `experiments/window_20261013T124402Z/`.

## Verification audit and windows

The audit checked NLI input order, label mapping, tokenization, truncation, softmax, neutral gating and output limits. It found no swapped-label or input-order bug. The pretrained model often assigned neutral to gold evidence.

Window experiments used up to three consecutive abstract sentences, stride two, with the title prepended. They compared all windows with the three windows having the highest lexical overlap with the claim. Inference windows never used gold sentence indices. Cached predictions supported candidate-depth and threshold comparisons. Aggregation used either the window with the highest non-neutral confidence or mean probabilities.

Two-stage calibration separated evidence confidence, P(SUPPORT)+P(CONTRADICT), from conditional label confidence, max(P(S),P(C))/(P(S)+P(C)). The grid used evidence thresholds 0, .01, .05, .1, .2, .3, .4, .5, .7 and .9; label thresholds .5, .7 and .9; and available candidate depths 1, 3, 5 and 10. Output was capped at three pairs. Validation exact-triple F1 determined selection, with precision used in tie-breaking.

## NLI fine-tuning

Both runs started from the same DeBERTa NLI model. Training used annotated SUPPORT/CONTRADICT pairs plus assumed-neutral samples. Every neutral pool excluded the claim's complete gold set, though unannotated relevant papers could still be sampled. Evidence-bearing claims supplied uniform and mined nongold samples; NEI claims supplied a mined sample.

One run used full abstracts. The other used the annotated training evidence span plus one neighboring sentence on each side for positive examples; neutral examples used lexical windows. This changed both training and inference inputs, so it was not a pure truncation comparison. Validation and evaluation never used gold sentences to choose windows.

Training used one epoch, AdamW 2e-5, weight decay .01, effective batch 8, gradient clipping 1.0, 10% warmup and float32. Two initial attempts ran out of MPS memory. They were restarted from the original weights using microbatch 2 with accumulation 4 and fixed 512-token padding. Failed attempts remain in the logs and are not counted as completed runs.

## Fusion and reranking

Cached rankings first supported an RRF sweep. A later score-fusion search compared six normalization methods across 45 weight combinations, giving 270 settings. The selected rule used the top-100 score spread without clipping, with weights BM25=.375, MedCPT=.375 and MiniLM=.25. Rankings were independently recomputed before adoption. Bootstrap and fold checks were descriptive checks on the same validation data, not a claim of global optimality.

MedCPT reranking used `(claim, title + abstract)`, a 512-token limit and raw relevance logits. Top-20 and top-50 cutoffs were compared on the old hybrid and the selected fusion. The submitted reranker uses the best eligible first stage and preserves its tail.

## Interruption and recovery

The machine shut down during a later run; its cause was not established. Severe memory pressure had been observed. On resumption, 4,028 validation scores were available on disk and no experiment model process remained. Subsequent inference used one workload, fixed 512-token shapes, batch 2, per-batch cleanup and an MPS memory fraction of .4 with watermark protection retained. Scores were saved every 32 pairs/windows.

The selected result was the full-abstract fine-tuned verifier over the final reranked candidates. Its validation F1 is 0.5827. See `FINAL_RESULTS.md` for the submitted results and `final-category-review.md` for the completed error analysis.
