# Final validation error-category decisions

The author asked an assistant to make the final category decisions on 20 October 2026, with a second assistant providing an independent check. The author did not review each case individually. This file and `final-category-decisions.json` contain the final decisions; earlier proposals are historical.

## Review method

The review checked all 74 error records against gold/predicted pairs, evidence sentences, extra-document abstracts, candidate ranks and sentence visibility; inspected complete classifier premises and probabilities for disputed cases. It also compared the decisions with the independent audit and applied the review criteria to the evidence. Categories describe observed error types, not demonstrated model-internal causes. Official gold labels and model predictions are unchanged.

When assigning one category per claim: required gold evidence outside the actual verifier pool takes precedence; otherwise use a specific observable semantic mismatch when supported. Scalar direction comparisons are numerical; causal activation/inhibition reversals and absence statements can be negation. Mixed, indirect, truncated or annotation-sensitive cases remain other. Multiple gold documents alone do not establish needs_multiple_docs. A finalized category may still have interpretive uncertainty.

## Corrections

- **6: other → entity_mismatch.** The extra bed-sharing paper measures bed-sharing-associated SIDS risk, not the proportion of SIDS deaths before six months. Outcome, population denominator and age comparison do not match; no arithmetic failure is established.
- **313: other → entity_mismatch.** The extra paper reports contig count and between-species noncoding-sequence lengths, not assembled-versus-unassembled contig lengths. It substitutes different objects and comparisons.
- **98: negation → numerical.** The fully visible result says apo(a) fractional clearance rate increased 24.6%; the claim says it decreased and the same document was labeled SUPPORT. Scalar direction-of-change error; later truncation does not remove this decisive sentence.
- **112: negation → other.** The visible paper describes both renal benefits and functional renal insufficiency as an adverse effect. SUPPORT and CONTRADICT scores nearly tie and the model abstains; a specific missed negation is not established.
- **645: negation → other.** The visible evidence links low folate/B6 intake to high homocysteine. Applying this inversely to the intake claim requires inference; the model predominantly predicts neutral. No explicit missed negation is demonstrated.
- **566: negation → numerical.** The visible evidence says deficient cells are hypersensitive to high iron, opposite the claimed reduced sensitivity. The model favors SUPPORT but abstains at the conditional threshold. The distinguishing error is scalar direction, not the shared absence-of-pathway condition.
- **523: other → numerical.** Although the annotated snippets emphasize mice, the full visible abstract explicitly reports that plasma L-carnitine predicts increased cardiovascular-event risk in 2,595 human subjects with high TMAO. This opposes the claimed negative correlation. The verifier abstained on this direction comparison; species mismatch is not the appropriate diagnosis.

Also corrected explanations for claims 61, 669, 673, 1115, 189, 537 and 1261 without changing their categories. In particular, the extra ART and LDL papers contain substantively relevant assertions; the extra KLF4 paper mentions KLF4 but addresses a different process.

## Final counts

- entity_mismatch: 11
- needs_multiple_docs: 0
- negation: 3
- numerical: 5
- other: 51
- retrieval_miss: 4

Four of 74 errors (5.41%) involve required gold documents outside the top-five verifier pool; all are inside top 100. No cross-document-combination requirement was established.

## All final decisions

| Claim | Category | Reason |
|---|---|---|
| 6 | entity_mismatch | The extra bed-sharing paper measures bed-sharing-associated SIDS risk, not the proportion of SIDS deaths before six months. Outcome, population denominator and age comparison do not match; no arithmetic failure is established. |
| 34 | other | Gold folate trial was sent to NLI but missed; a different folic-acid trial was labeled SUPPORT and may be relevant but is unannotated. |
| 55 | other | Gold AMPK/metformin lung-fibrosis paper reached NLI and states lower fibrotic activity, yet the verifier abstained. |
| 61 | other | The gold ART paper is correct. The extra financing paper explicitly says ART reduces HIV transmission; its extra prediction is annotation-sensitive, not demonstrably unrelated merely because its main topic is financing. |
| 63 | other | Gold NEI; ATR checkpoint article plausibly supports DNA-damage sensing, exposing a possible annotation gap rather than a retrieval miss. |
| 64 | other | Gold NEI; ATM/ATR papers plausibly contradict the no-role claim, but these documents are not annotated evidence. |
| 88 | other | Gold macrophage self-maintenance paper reached NLI but was missed; two alternative macrophage papers were labeled SUPPORT. |
| 98 | numerical | The fully visible result says apo(a) fractional clearance rate increased 24.6%; the claim says it decreased and the same document was labeled SUPPORT. Scalar direction-of-change error; later truncation does not remove this decisive sentence. |
| 106 | entity_mismatch | Extra contradiction papers concern ambulatory blood-pressure treatment/profiles, not accuracy of hypertension diagnosis; one gold diagnostic paper was missed. |
| 112 | other | The visible paper describes both renal benefits and functional renal insufficiency as an adverse effect. SUPPORT and CONTRADICT scores nearly tie and the model abstains; a specific missed negation is not established. |
| 119 | other | Gold migraine review was correct; extra amitriptyline paper also discusses migraine prophylaxis but is not annotated. |
| 155 | other | Gold BCL-2 paper reached NLI but was rejected; annotated sentence discusses effects of BCL-2 elimination rather than the claim phrasing. |
| 167 | other | Gold bariatric-surgery meta-analysis was correct; extra postoperative-psychological-outcome paper was labeled SUPPORT. |
| 189 | negation | The visible sentence says 10E8 did not bind phospholipids, corresponding to the gold no-affinity statement; the model abstains. Classify the observable absence-of-binding relation as negation. The claim spells the antibody 10EB, an additional uncertainty; this category does not prove a model-internal cause. |
| 235 | other | Gold chicken study states somatic sex identity is cell autonomous; verifier abstained. |
| 256 | other | Gold age-specific prevalence evidence reached NLI, but was rejected; a specifically numerical cause is not established. |
| 264 | other | Gold NETosis paper links citrullinated antigens, autoantibodies and inflammation, but verifier assigned the opposite label. |
| 296 | entity_mismatch | Gold cyclin-A destruction paper was correct; extra CONTRADICT paper concerns Cyclin A2 control of Mre11 translation, a different mechanism. |
| 313 | entity_mismatch | The extra paper reports contig count and between-species noncoding-sequence lengths, not assembled-versus-unassembled contig lengths. It substitutes different objects and comparisons. |
| 341 | other | Gold surgery-versus-conventional diabetes-remission trial was correct; extra bariatric-surgery cohort was also marked CONTRADICT. |
| 349 | other | Gold ovarian-reserve paper reached NLI; several evidence sentences were truncated and the classifier abstained. A numerical cause cannot be isolated. |
| 355 | other | Both gold Drosophila boundary papers were sent to NLI but missed; an unannotated imaginal-disc actomyosin paper was labeled SUPPORT. |
| 375 | entity_mismatch | Gold cell-free mtDNA mortality paper was correct; extra CONTRADICT paper discusses mitochondrial DNA in neutrophil traps, not mortality. |
| 381 | other | Gold gene-environment paper was correct; two alternative breast-cancer papers on prenatal environment/birthweight were labeled SUPPORT. |
| 382 | other | Gold progerin-induced aging study reached NLI but was rejected; the claim paraphrases progerin expression broadly as adding molecules. |
| 413 | other | Gold APOE4/dementia paper reached NLI but was rejected; annotated association depends on reproductive-period strata and text visibility is limited. |
| 417 | other | Gold reproductive-period paper was rejected; its annotated sentence addresses dementia risk among carriers rather than directly proving shorter estrogen exposure. |
| 430 | other | Gold FOXO paper reached NLI but was rejected; annotated sentence concerns Sir2 tipping FOXO away from apoptosis, not direct ROS inhibition. |
| 463 | other | Gold metastasis-model paper was missed after NLI; three extra papers on molecular profiling/CTCs/ctDNA were labeled SUPPORT. |
| 483 | other | Gold H. pylori mutation paper was correct; extra mouse gastric-mutation study is related but unannotated. |
| 490 | other | Gold NEI; HNF4A article studies macrosomia but supplied abstract gives no result establishing the association. |
| 523 | numerical | Although the annotated snippets emphasize mice, the full visible abstract explicitly reports that plasma L-carnitine predicts increased cardiovascular-event risk in 2,595 human subjects with high TMAO. This opposes the claimed negative correlation. The verifier abstained on this direction comparison; species mismatch is not the appropriate diagnosis. |
| 531 | retrieval_miss | Two annotated embryonic-stem-cell documents rank 8 and 11, beyond the top-5 NLI pool; a different unannotated supporting document was emitted. |
| 537 | negation | Visible evidence says neuronal activation is necessary for panic and silencing/antagonism blocks it, whereas the claim says the neurons suppress panic. The same document is labeled SUPPORT. Negation category covers this reversal of activation versus inhibition of the causal intervention. |
| 541 | other | Both gold hypothalamic-glutamate papers were predicted correctly; extra astrocyte-glutamate paper was also labeled CONTRADICT. |
| 559 | retrieval_miss | Only annotated spatial-segregation paper ranks 54 and never reached the top-5 NLI verifier. |
| 566 | numerical | The visible evidence says deficient cells are hypersensitive to high iron, opposite the claimed reduced sensitivity. The model favors SUPPORT but abstains at the conditional threshold. The distinguishing error is scalar direction, not the shared absence-of-pathway condition. |
| 594 | numerical | Gold heart-failure cohort reports no change over time in women, contradicting the claimed 10% increase; verifier abstained. |
| 616 | other | Gold NEI; retrotransposon papers describe insertional mutagenesis/cancer, making the no-mutation claim plausibly contradicted but unannotated. |
| 645 | other | The visible evidence links low folate/B6 intake to high homocysteine. Applying this inversely to the intake claim requires inference; the model predominantly predicts neutral. No explicit missed negation is demonstrated. |
| 669 | entity_mismatch | The gold differentiation paper is correct. The extra paper does mention KLF4, but its necessity concerns self-renewal/proliferation of already differentiated MafB/c-Maf-deficient cells, rather than the claimed requirement for myeloid differentiation. Biological-process mismatch. |
| 673 | other | The gold paper is correct. One extra paper explicitly calls LDL a causative agent of coronary disease; the other discusses lipid treatment and residual risk. These are unannotated extra predictions; an incorrect causal direction is not established. |
| 682 | other | Gold FGF21-deficiency paper reached NLI and reports premature death in mice; verifier abstained. |
| 738 | retrieval_miss | One annotated MICAL paper ranks 22, beyond top-5 NLI, so exact gold-set recovery was impossible even though other gold papers were examined. |
| 751 | other | Gold MVP/miR-193a paper explicitly links exosomal sorting to tumor progression; verifier abstained. |
| 761 | other | Gold NEI; three MeCP2 papers on dendritic/spine/neuronal maturation were marked SUPPORT, suggesting an annotation-sensitive broad claim. |
| 766 | entity_mismatch | Gold obesity-drug adverse-effects paper was correct; extras discuss olanzapine (antipsychotic) and diabetes/cancer drugs rather than anti-obesity medication. |
| 780 | other | Two gold IFN-gamma-deficiency myocarditis papers reached NLI and report severe disease, but verifier abstained on both. |
| 815 | other | Gold RIM1 paper explicitly says rim1 mutations reduce IME1 expression; verifier abstained. |
| 825 | other | Gold N348I/nevirapine paper was correct; extra mechanistic N348I paper also supports resistance but is unannotated. |
| 886 | entity_mismatch | The gold survey was correctly labeled CONTRADICT; the error is an extra SUPPORT prediction from a sciatica-treatment trial, which does not establish surgical-trial discontinuation rates. |
| 895 | entity_mismatch | Gold NEI; predicted paper studies malaria retinopathy in Bangladeshi adults, not Gabonese children with SFM or plasma lactate. |
| 898 | other | Gold NEI; two oxidized-DNA papers describe STING-dependent sensing, so support predictions may reflect incomplete annotation. |
| 954 | entity_mismatch | Gold pioglitazone/prostate-cancer paper was correct; extra paper concerns bladder-cancer risk, a different cancer site. |
| 962 | other | Two gold mitochondrial-RNA-granule papers were correct; one gold DDX28 paper was missed and an unannotated RNA-granule paper was emitted. |
| 984 | entity_mismatch | Gold PTENP1 pseudogene paper was correct; extra paper concerns PTEN pathway activity and prognosis, not PTENP1 regulation. |
| 1003 | retrieval_miss | An annotated RAD52/BIR paper ranks 6, beyond top-5 NLI; other gold papers were also missed and an SSA paper was mislabeled. |
| 1009 | other | Gold TLX1/T-ALL article suggests RUNX1 tumor-suppressor role, but the claim adds downregulation or mutation not directly in annotated text; verifier abstained. |
| 1028 | other | First gold IL2RA/Treg paper was correct; second gold IL-2 reduction/Treg paper was sent to NLI but rejected. |
| 1073 | numerical | Gold sepsis study reports declining in-hospital mortality and no significant death-or-hospice trend, opposite a rise from 2009–2014; same paper mislabeled SUPPORT. |
| 1115 | other | The decisive no-significant-difference result is visible despite later truncation. The model abstains with nearly tied polarity scores. Statistical non-significance versus the broad no-superiority claim and threshold uncertainty prevent isolating a specific negation failure. |
| 1125 | other | Gold ocular-inflammation study finds most immunosuppressants do not raise cancer mortality but mentions a TNF-inhibitor exception; verifier abstained. |
| 1148 | other | The gold sentence containing the 10% statistic was not visible to the classifier because of truncation; this is an input-coverage failure, not demonstrated numerical reasoning failure. |
| 1223 | other | Gold malaria paper says innate restriction saturates with dose; inference that small doses are controlled was not accepted. |
| 1235 | other | Gold NEI; brown-fat review says browning increases energy expenditure, plausibly contradicting the decreased-potential claim. |
| 1246 | other | The gold mortality statistics require contextual interpretation of the broad hospitable-environment claim; the classifier abstained, with no isolated numerical error established. |
| 1249 | other | Gold childhood-fracture study places incidence peak at puberty, not toddlerhood; verifier abstained. |
| 1260 | other | Gold CYP2D6 outcome paper was rejected; extra genetic-polymorphism paper concerns tamoxifen metabolite levels, not treatment outcome directly. |
| 1261 | negation | Visible evidence says oncogene-induced ROS causes DNA damage, whereas the claim attributes instability to ROS removal. The same paper is labeled SUPPORT. Negation category covers reversal of production versus removal in this causal chain. |
| 1291 | other | Gold NEI; HNF4A article identifies MODY gene but supplied abstract has no diabetes-risk comparison, so contradiction overreaches. |
| 1311 | other | Gold lincRNA-p21 translation paper reached NLI and describes trans-acting regulation, yet verifier abstained. |
| 1324 | other | Gold fibronectin study says FN reconstitution restores aged-muscle regeneration; verifier abstained. |
| 1390 | other | Gold modified-tRNA paper was correct; extra yeast wobble-nucleoside paper was labeled SUPPORT but is unannotated. |
| 1404 | other | One gold A20 xenograft paper was correct; second gold A20-targeting xenograft paper reached NLI but was rejected. |

## Reconciliation with GPU manager review

Checked the completed GPU manager review in chat 01a13696-1d41-7442-98e8-e15681948f25 and its independent-category-audit-2026-10-20.md. All five recommended category changes are accepted: 6 and 313 to entity_mismatch; 98 to numerical; 112 and 645 to other. Its proposed explanation corrections for 61 and 669 are accepted. These were already incorporated in the final decisions above.

Resolved its open choice for 566 as numerical: the contrasting quantity is sensitivity (hypersensitive versus reduced sensitivity); both claim and evidence refer to pathway deficiency.

One evidence-based departure: retain our correction of 523 to numerical, despite the independent audit recommending other. The complete, untruncated classifier input explicitly reports increased cardiovascular-event risk in 2,595 human subjects with high TMAO. The audit's mouse-only rationale reflects the annotated snippets rather than the full input. The negative-correlation claim conflicts with that visible human result.

Final counts remain 51 other, 11 entity_mismatch, 5 numerical, 3 negation, 4 retrieval_miss, 0 needs_multiple_docs. The staged CSV exactly matches all 74 final decisions. No further category changes were needed on reconciliation.
