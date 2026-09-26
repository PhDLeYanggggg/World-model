# Auxiliary Cost Failure Analysis

## Observed, Not Hypothesized
1. Fitting executed normally. All432 heads completed their fixed budget;
   unknown-label draws0 and288 initialization/sampler/preprocessing matches
   passed. The true auxiliary's fixed-batch cost loss fell in144/144 heads,
   versus143/144 for each control. Loss descent does not imply convergence.
2. Full-input cost-ranking AUROC improves against both matched controls in
   all six assignment intervals. The task is not wholly unlearnable.
3. Expected easy-harm cost MSE fails against the strong original:0 favorable,
   3 adverse and3 overlapping intervals. The auxiliary does not consistently
   beat shuffled labels either. This rejects the registered repair, not all
   possible event-aware methods.
4. Full-input true-versus-cost-only median fitting MSE gain is1.309%, versus
   held median0.422% across72 dependent views. There are19 views with improved
   fitting but worse held MSE. Against shuffled, the corresponding medians
   are2.789% and0.267%, with24 reversals. These are descriptive summaries, not
   additional independent inferential tests.
5. Motion-only has34/72 fit-positive/held-negative views against cost-only.
   Its held median gain is-0.0497%, despite fitting median+0.7681%. Sparse
   event support remains; three held event rankings are not_estimable.

## Failure Taxonomy

| Candidate explanation | Evidence status | Consequence |
|---|---|---|
| Runtime or failed optimization execution | Not supported by completion/loss checks | Do not blame the old OpenMP issue |
| No causal event signal | Contradicted for full-input ranking | Distinguish occurrence/ranking from magnitude |
| Generalization/transport instability | Fit/held reversals and adverse locality intervals observed | Inspect support and magnitude transport before enlarging models |
| Auxiliary task interference | Possible; cost gains vary and shuffled can win | Not uniquely identified by present controls |
| New base estimator weaker than original | Directly observed; cost-only has two adverse primary intervals against original | Reconstruct a genuinely matched strong control |
| Original-versus-new architecture/input/objective confounding | Known by design | Cannot attribute all degradation to the auxiliary label |
| Two-to-three-locality producer transport | Unresolved design limitation | Audit without using outer outcomes as training labels |
| Too few updates or wrong weight | Untested explanations | Do not claim longer training or a weight sweep will fix it |
| Independent-scene generalization | not_run | No confirmation claim |

## Diagnostic Implementation Repair
The first secondary fit/held report stopped on a legacy-format mismatch:
the original fitting artifact stores all-row MSE, not positive-envelope MSE.
It produced no completed diagnostic. The reader now preserves those72
fitting comparisons per family as unavailable instead of mixing denominators.
A regression test reproduces this case. Training, primary held readout and
registered gates were unchanged; all held comparators remain available.

## Next Falsifiable Action
First reproduce the original cost estimator under its exact causal inputs,
architecture, four-cost objective, support/sampling and budget. Then any
auxiliary test must alter only the auxiliary supervision, retain an exact
no-auxiliary control and preserve the original estimator as a reference.
Before choosing a new loss, describe fitting-only signed magnitude errors
and source support by locality. A narrow repair must improve the strong
control's cost accuracy and tail guards, not only AUROC.

No outer-error-driven thresholds, best-scene selection or promotion of the
favorable2/6 comparison. Unknown support is not zero error. Full/motion is
not a matched feature ablation. The source pool has prior development
exposure. Obs8/pred12 and detector pixels only; no safety/metric/seconds/
true3D/foundation claims. No Stage5C, SMC or deployment changes.
