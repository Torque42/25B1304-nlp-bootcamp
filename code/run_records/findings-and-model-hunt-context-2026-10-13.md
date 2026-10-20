# Early experiment results — 13 October 2026

This is a historical summary of the experiments that led to the submitted MiniLM/MedCPT/DeBERTa configuration. It has been edited for readability from the original working notes. It is not a current work queue. The numerical run records remain unchanged.

The submitted system reaches validation retrieval nDCG@10 0.9108547394 and verification F1 0.5826771654. The higher verification score of 0.5982905983 below used the old hybrid and a different candidate policy; it is not the submitted pipeline's score.

## Recorded comparisons

### BM25: eight configurations

Title+abstract vs abstract only × k1{0.9,1.5} × b{0.4,0.75}.

| Text | k1 | b | nDCG@10 |
|---|---:|---:|---:|
| title+abstract | .9 | .4 | .836567 |
| title+abstract | .9 | .75 | .835362 |
| title+abstract | 1.5 | .4 | .824014 |
| title+abstract | 1.5 | .75 | .827221 |
| abstract | .9 | .4 | .823039 |
| abstract | .9 | .75 | .816815 |
| abstract | 1.5 | .4 | .820518 |
| abstract | 1.5 | .75 | .816794 |

### Dense retrieval and Task 4

| Model/run | nDCG@10 |
|---|---:|
| Pretrained MiniLM | .8155492691 |
| Pretrained MedCPT | .8508920087 |
| MiniLM random-negative FT | .8247126786 |
| MiniLM hard-negative FT | .8395772959 |

Both Task 4 runs started from identical pretrained weights and used 440 positive pairs. Each ran for one epoch (28 steps), with batch size 16, AdamW at 2e-5 and contrastive temperature .05. Each positive had one negative; there were no in-batch negatives. Random negatives were uniform nongold samples. Hard negatives came from the top 20 nongold candidates after fusing the three retrieval lists. All gold documents, including contradiction evidence, were excluded from negative pools. Both runs changed the weights. Mean losses were .00468 for random negatives and .20556 for hard negatives. One epoch does not establish overfitting. MedCPT was not fine-tuned in this series.

### Hybrid fusion

- Equal-weight BM25 + MedCPT RRF with constant 60 scored .8833138923.
- The RRF sweep covered 100 settings, including constants 10, 30, 60 and 100; two-retriever weight ratios; and three-retriever weights in quarter increments. Some settings were equivalent.
- The best RRF setting used constant 10 and BM25/MedCPT/MiniLM weights .25/.5/.25, scoring .8910492916.
- Score fusion expanded from a 42-setting pilot to 270 settings: six normalization methods and 45 weight combinations. Both searches chose top-100 score-spread normalization with weights .375/.375/.25, scoring .8960605345.
- The pilot and expanded search overlap; they are not 312 independent experiments. Both reused cached embeddings and CPU scores without encoder training.

### Reranker experiments

| Reranker / first stage | k | nDCG@10 |
|---|---:|---:|
| MiniLM / original hybrid |20|.8773483118|
| MiniLM / original hybrid |50|.869463|
| MiniLM / original hybrid |100|.864847|
| MedCPT / original hybrid |20|.9191770297|
| MedCPT / original hybrid |50|.9125231857|
| MedCPT / selected fusion |20|.9108547394|
| MedCPT / selected fusion |50|.9091627405|

A further 15 cached diagnostic settings blended normalized first-stage scores with MiniLM reranker scores: three prefix depths (20, 50, 100) and five reranker weights (.1, .25, .5, .75, 1). The best scored .8916913181 at depth 100 and weight .1. This blend was not submitted because the assignment requires cross-encoder ordering in the reranked prefix. Neither reranker was fine-tuned.

### Verification experiments on the old hybrid

| Variant | Best F1 |
|---|---:|
| Original zero-shot full abstract |.3269230769|
| Zero-shot full, calibrated evidence/polarity |.4242424242|
| Zero-shot all windows, ordinary gate |.4210526316|
| Zero-shot all windows, max aggregation calibrated |.4416666667|
| Zero-shot all windows, mean aggregation calibrated |.4330708661|
| Zero-shot lexical windows, ordinary gate |.3819095477|
| Zero-shot lexical windows, max calibrated |.4069264069|
| Zero-shot lexical windows, mean calibrated |.4177777778|
| Fine-tuned full abstract, ordinary gate |.5836909871|
| Fine-tuned full abstract, calibrated |.5982905983|
| Fine-tuned evidence-span model, lexical windows ordinary gate |.5769230769|
| Same window model, mean aggregation calibrated |.5791855204|

The initial zero-shot sweep compared depths 5 and 10 with thresholds .4 to .9; depth 5 at .5 won. The best depth-10 F1 was about .279. Windows contained three sentences with stride two and the title prepended. Later comparisons used all windows or the top three by lexical overlap, candidate depths 1/3/5/10 where scored, and shared or separate evidence/polarity thresholds. They compared maximum-confidence and mean-probability aggregation.

Both training modes used 1,486 examples and the same one-epoch optimization budget. Evidence-window positives used the training evidence span plus neighboring sentences; negative and inference windows used lexical selection. This changes both training and inference inputs, so it is not a pure truncation ablation.

Finally, both trained variants were evaluated with the selected reranker, reusing compatible logits and scoring missing pairs. The full-abstract model at depth 5 won with F1 .5826771654.


## What the audits showed

The baseline NLI input order, label mapping, softmax and tokenizer were correct. The main observed problem was frequent neutral predictions on gold evidence. Among 114 gold pairs available at the old hybrid's top-five cutoff, the original model predicted 72 neutral, 36 correct labels and six wrong labels. After full-abstract fine-tuning, the corresponding counts were 18, 78 and 18. Fine-tuning reduced abstention but did not eliminate label errors.

The original hybrid contained 83, 110, 114, 120 and 124 of the 124 gold pairs at depths 1, 3, 5, 10 and 20. Selected fusion contained 86, 105, 116, 122 and 122. A better nDCG score can therefore coexist with lower evidence coverage at a particular downstream cutoff.

Training examples were checked for split separation, and assumed-neutral pools excluded each claim's known gold documents. The neutral labels remain assumptions: a nongold paper can still be relevant.

## Limits of the search

The experiments used discrete grids, not binary search; quality was not monotonic in these settings. Fusion weights and decision thresholds received more attention than learning rates, losses, model families or training duration. The selected settings are the best of those compared, not proven optima.

On the same validation data, descriptive paired confidence intervals included zero for the RRF gain (about .00774; interval [-.01663, .03287]) and score-fusion gain (about .01275; interval [-.01049, .03624]). Repeated fold selection also reduced the apparent gains. Fine-tuning showed a larger improvement, though its validation-based selection still introduces optimism. The reranker comparison changed both first stage and cross-encoder, so its gain is not a clean model-only ablation.

These observations motivated later model exploration. They do not establish evaluation-set performance. The final submission uses the configuration and metrics in `FINAL_RESULTS.md`.
