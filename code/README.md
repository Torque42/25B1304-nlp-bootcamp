# Scientific claim retrieval and verification

This project retrieves abstracts for a scientific claim, reranks them, and predicts whether they support or contradict it. It includes the seven retrieval runs and verification outputs required by the assignment.

All commands below run from the extracted submission folder: the directory containing `code/`, `predictions/`, `results.json`, and `report.md`.

## Start here

- **To inspect the submission:** read `report.md`, `results.json`, and `code/run_records/FINAL_RESULTS.md`.
- **To check the supplied files:** follow “Check the submitted outputs.” This uses the Python standard library and does not run a model.
- **To recreate the experiments:** follow “Reproduce the pipeline” in a separate extracted copy. It retrains models and replaces the supplied outputs.

## What is included

```text
code/                  Scripts, pinned dependencies, tests and saved run records
predictions/           Nine TREC files and three verification/error CSVs
results.json           Settings, measured results and review status
report.md              Answers to Q1-Q5
```

The original dataset and model weights are not included. Obtain the four JSONL files from the assignment and place them in `data/` alongside `code/`:

```text
data/corpus.jsonl          5,183 abstracts
data/train_claims.jsonl      647 claims
data/val_claims.jsonl        162 claims
data/eval_claims.jsonl       300 claims
```

Keep the original IDs and annotations. The checker and results template are already bundled in `code/`; no PDF, Git checkout, or files from the author's machine are needed.

## Check the submitted outputs

Python 3.12 is recommended. No packages need to be installed for this check:

```sh
python3 -B code/validate_submission.py --staging . --data data \
  --val-run scientific=code/run_records/final_validation_runs/scientific.trec \
  --val-run hybrid=code/run_records/final_validation_runs/hybrid.trec \
  --val-run rerank=code/run_records/final_validation_runs/rerank.trec \
  --val-run random=code/run_records/final_validation_runs/random.trec \
  --val-run hard=code/run_records/final_validation_runs/hard.trec
```

A successful check prints `"status": "PASS"` and exits with code 0. It checks IDs, coverage, score ordering, the reranked prefix and unchanged tail, all seven validation nDCG values, verification precision/recall/F1, error-category coverage, report lengths and review records. The extra validation runs above belong to the submitted experiment; after a fresh reproduction, use its newly generated rankings to check its results.

For the unmodified assignment checker:

```sh
python3 -B - <<'PY'
import sys
sys.path.insert(0, 'code')
import check_format
checked = check_format.check_submission('.', check_format.load_data_info('data'))
print(checked['status'])
print(checked['general'])
assert not checked['general']
assert all(status == 'OK' for status in checked['status'].values())
PY
```

This explicit data path matters: running `python code/check_format.py .` directly would make that checker look for `code/data/`.

## The submitted configuration

| Task | Model and settings |
|---|---|
| 1: lexical retrieval | BM25 on title + abstract; k1=0.9, b=0.4 |
| 2: general retrieval | Pretrained `sentence-transformers/all-MiniLM-L6-v2` |
| 2: scientific retrieval | Pretrained `ncbi/MedCPT-Query-Encoder` and `ncbi/MedCPT-Article-Encoder` |
| 3: fusion | BM25 0.375, MedCPT 0.375, MiniLM 0.25; query-specific top-100 min-max normalization, without clipping |
| 3: reranking | `ncbi/MedCPT-Cross-Encoder`; rerank the top 20 and keep ranks 21-100 in first-stage order |
| 4: retrieval fine-tuning | Two independent MiniLM runs, using random and hard negatives |
| 5: verification | Fine-tuned `cross-encoder/nli-deberta-v3-small`; full abstracts; top 5 candidates; evidence threshold 0.9 and conditional label threshold 0.7; at most 3 evidence pairs |

Task 3 uses the pretrained MiniLM embeddings. Task 4's two fine-tuned models are separate retrieval runs. Task 5 uses Task 3 reranking, which had the highest validation nDCG@10. Exact pretrained revisions are in `code/config.json` and `code/selected_recipe.json`. The final verifier checkpoint and thresholds are recorded in `code/run_records/nli_selection.json`.

For fusion, each retriever's scores are normalized as:

```text
(score - score_at_rank100) / max(top_score - score_at_rank100, 1e-12)
```

Scores below the rank-100 score may be negative. Reranker inputs are `(claim, title + abstract)`, truncated to 512 tokens. Raw model logits determine the prefix order; the TREC file uses `101 - rank` as an order-preserving score. The verifier uses `(title + abstract, claim)`, with only the abstract premise truncated when the pair exceeds 512 tokens.

For NLI probabilities S=SUPPORT and C=CONTRADICT, evidence is accepted when `S+C >= 0.9` and `max(S,C)/(S+C) >= 0.7`. The larger probability determines the label. If more than three pairs pass, keep the three with the highest chosen-label probability, breaking ties by retrieval rank and then document ID. No output rows for a claim means NOT ENOUGH INFO. Input order, pooling, token limits and label mappings are implemented in `neural.py`, `safe_inference.py`, and the experiment scripts.

## Reproduce the pipeline

Use a fresh extracted copy, Python 3.12 and an internet connection for public Hugging Face downloads. No account token is required for the selected public checkpoints. Install the pinned environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r code/requirements.txt
```

The original runs used macOS with PyTorch MPS, float32 and seed 42. Confirm that MPS is available before choosing it:

```sh
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available())"
```

Use `--device cpu` in place of `--device mps` if this prints `False`. CPU execution is supported, but its full runtime has not been measured. CUDA is not exposed by the selected runner. Hardware and library differences can affect floating-point results, and seeded MPS execution is not guaranteed to be bitwise identical.

The complete command sequence is:

```sh
.venv/bin/python -B code/run_selected.py --device mps --through-stage materialize
.venv/bin/python -B code/submission.py assemble
```

The first command recreates the baseline, applies the recorded fusion recipe, trains both NLI controls, compares reranker cutoffs and verifier settings on validation, then generates evaluation predictions. The second writes results, draft answers and review material from those new measurements. It does not preserve the edited answers supplied in this archive. Since selection is repeated, a numerically different run can choose different settings; inspect the new `artifacts/nli_selection.json` before comparing outputs.

To run or resume one stage at a time, use these commands in order:

```sh
# Tasks 1-4 and the initial Task 5 baseline
.venv/bin/python -B code/run_selected.py --device mps --from-stage baseline --through-stage baseline
# Selected fusion recipe
.venv/bin/python -B code/run_selected.py --device mps --from-stage fusion --through-stage fusion
# Full-abstract and evidence-window NLI training controls
.venv/bin/python -B code/run_selected.py --device mps --from-stage nli_training --through-stage nli_training
# MedCPT reranker comparison
.venv/bin/python -B code/run_selected.py --device mps --from-stage reranking --through-stage reranking
# Verifier selection on validation, then evaluation inference
.venv/bin/python -B code/run_selected.py --device mps --from-stage verification_selection --through-stage materialize
.venv/bin/python -B code/submission.py assemble
```

Dependencies must exist before a later stage starts. `code/run.py` runs only the initial baseline. Always specify `--through-stage materialize` for the complete selected run: the runner's default final stage creates a draft package.

Working files go in `artifacts/`; model downloads and score caches are kept there too. Resume can reuse compatible checkpoints and scores, but a saved checkpoint does not mean its validation scoring finished. Failed training without a complete checkpoint restarts that training run. Reranker/NLI scores are saved periodically. Do not mix artifacts from different configurations or copy historical `run_records/` into `artifacts/` to bypass computation. If cache identity checks fail, use a fresh working copy rather than editing the cache metadata.

Run one neural workload at a time. The scripts use conservative inference batches and MPS memory guards. They do not enforce a two-hour cooling schedule; for long laptop runs, schedule breaks between stages and stop at a safe stage boundary before putting the laptop away. Avoid disabling PyTorch's memory protections.

## Training details

Task 4 has 440 positive training claim-document pairs. SUPPORT and CONTRADICT evidence both count as retrieval positives. Each pair gets one negative, excluding every annotated gold document for that claim. Random negatives are uniform nongold samples. Hard negatives are sampled from the top 20 nongold candidates after equal-weight RRF (constant 60) over BM25 and both pretrained dense top-100 lists.

Both MiniLM runs start from identical pretrained weights: one epoch, batch 16, AdamW at 2e-5, weight decay 0.01 excluding bias/LayerNorm, temperature 0.05, two-candidate cross-entropy, no in-batch negatives, 10% warmup followed by linear decay, and gradient clipping at 1.0. Initial/final weight hashes and losses are saved.

The selected NLI model is trained for one epoch on 440 labeled evidence pairs and 1,046 assumed-neutral nongold pairs. Those neutral labels may contain annotation noise. It uses AdamW at 2e-5, effective batch 8 through microbatch 2, fixed padding to 512, 10% warmup/linear decay and clipping at 1.0. The evidence-window training control uses training evidence annotations for its positive premises; validation/evaluation window selection does not use gold sentence indices. The full-abstract model was selected for this submission.

## Reading the results

All reported scores use the supplied validation set. Retrieval metrics average the 106 claims with gold evidence; verification includes all 162 claims. Validation was also used to choose settings and thresholds, so these scores are development measurements. Evaluation predictions were produced after selection; evaluation labels were not used by this selected pipeline.

Verification compares exact `(claim, document, label)` triples. A wrong label contributes both a false positive and a false negative. An error-category row is required when a claim's complete predicted pair set differs from gold.

Index time includes corpus tokenization, encoding and matrix construction, excluding model loading/download. Query latency uses 30 sorted validation queries after warmup, with MPS synchronization. Hybrid and reranker totals add separately measured stage means. Memory is peak process RSS in MiB; MPS allocation is recorded separately and can overlap it on unified-memory machines.

Final category decisions were made by an assistant at the author's explicit request and checked by a second assistant. `code/run_records/final-review-provenance.json` records that process. `final-category-decisions.json` and `predictions/t5_val_errors.csv` contain the final labels. Older blank review sheets are retained as experiment history. Machine-generated logs and measurements are unchanged.

## Reviewing a fresh run

A fresh run can change predictions and therefore the error set. Open `artifacts/validation_error_review.html`, review every incorrect claim, and export the category CSV. Import it using:

```sh
.venv/bin/python -B code/verification.py --import-review /absolute/path/to/reviewed_categories.csv
.venv/bin/python -B code/submission.py assemble
```

Then review the regenerated answers. Do not copy the submitted categories onto changed predictions. `submission.py assemble` and `submission.py package` rebuild draft report/review material, so do not call them merely to archive answers you have already edited.

To archive a finalized tree without regenerating results, run the standalone validator with `--write-zip ../25B1304_nlp_bootcamp.zip` added to the full validation command above. Keep the destination outside the submission folder. The validator refuses to overwrite an existing file and includes only `code/`, `predictions/`, `results.json`, and `report.md`. Local `data/`, `artifacts/`, `.venv/`, Git metadata and Python caches are excluded.

## Small CPU tests

After installing the dependencies, run the metric, ranking, sampling, review-import and resume tests without model downloads:

```sh
.venv/bin/python -B -m pytest -q -p no:cacheprovider code/tests
```

## What was checked before submission

The format checker passed all 12 prediction files. All seven retrieval metric triples and verification P/R/F1 were independently recomputed. Q1 ranks, training hashes, final categories, reranker-tail preservation and report word limits were checked. The packaged command interfaces and local validation workflow were also tested. A full fresh neural training/inference run was not repeated during packaging; exact bitwise reproduction is not claimed.
