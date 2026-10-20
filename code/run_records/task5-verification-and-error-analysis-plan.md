# Verification and error analysis

## What the pipeline does

For each claim, the retrieval system supplies candidate abstracts. An encoder NLI model assigns support, contradiction and neutral scores. The selected decision rule accepts up to three evidence pairs. A claim with no accepted pair is treated as NOT ENOUGH INFO.

Training uses the provided training claims. Validation is used for model selection, threshold calibration, measurement and error analysis. Evaluation inference uses the selected settings. The retrieval and verification predictions come from encoder models; the final error categories were reviewed separately by assistants at the author's explicit request.

## Saved evidence for the analysis

The review records contain each claim, its gold and predicted pairs, missing and extra pairs, document ranks, evidence sentences, the text visible to NLI, probabilities and decision thresholds. These records distinguish an abstract that was never considered from one that reached NLI and was rejected.

Nested records use JSONL; CSV and HTML views make them easier to inspect. Evaluation traces contain no gold labels. The final decisions are in `final-category-decisions.json`, and `final-review-provenance.json` records how the review was performed.

## Scoring and categories

Verification precision, recall and F1 compare exact (claim, document, label) triples. For the error file, a claim is incorrect if its complete predicted pair set differs from gold. Every incorrect claim gets one category; correct claims get none.

| Category | Criterion used in review |
|---|---|
| retrieval_miss | Required gold evidence did not reach the verifier, either outside top 100 or below the actual NLI cutoff. |
| negation | The error concerns an absence statement or a reversal such as activation versus inhibition. |
| numerical | A quantity, comparison or direction of change was misread. |
| entity_mismatch | The accepted evidence concerns a different intervention, condition, population, outcome or biological process. |
| needs_multiple_docs | Correct classification genuinely requires combining information across abstracts. |
| other | The available evidence does not justify a more specific category, including truncation, ambiguous inference, threshold rejection and annotation-sensitive extra predictions. |

These are practical review criteria, not proof of a model's internal reasoning. A number or negation word in a claim is not enough to assign the corresponding category. Multiple gold papers do not by themselves establish a multi-document reasoning requirement. For gold-NEI claims, an extra prediction is not a retrieval miss merely because there is no gold document.

When several failures occur in one claim, required evidence outside the candidate pool takes precedence. Otherwise the review uses a specific observable mismatch when supported, and `other` when the cause remains unclear. Gold annotations and model predictions were not edited during the review.

## Final checks

The final CSV has exactly the 74 incorrect validation claim IDs, each once, with an allowed category. Four claims are retrieval misses, all below the top-five verifier cutoff but within top 100: 4/74 = 5.41%. No case clearly required combining documents. The assignment checker verifies file structure; the separate audit checks exact error coverage and consistency with the recorded decisions.
