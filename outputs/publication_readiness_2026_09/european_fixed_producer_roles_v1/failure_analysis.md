# Failure Taxonomy And Follow-Up

## Resolved Engineering Error

The new preflight initially confused the CV evaluation reference with the frozen
forecaster's train-selected baseline index. Exact replay failed before training.
Checkpoint identity now supplies that original index; all9 forecasters replay.
No earlier checkpoint, prediction, threshold or historical metric was changed.

## Useful Repair: Preserve The Producer And Floor

The previous study replaced the stronger floor with two-source variants and
retained easy failures even when the learned increment helped. Here every arm
shares the original four-source floor. It has no easy locality/view violations.
Matched supervision has none either; OOF adds6 and ridge adds5. This supports
keeping a stable inference producer chain. It does not isolate producer identity:
training forecast quality, labels, cohort and normalizers change together.

## Remaining Prediction Failures

1. **Controller-reference mismatch.** The learned utility/risk objective improves
   on the floor, but the real deployment competitor is already a learned stopping
   controller. All3 negative all-ADE intervals arise when removed original switches
   lose more benefit than newly added switches gain. The algorithm can safely
   rediscover a weaker policy without beating the incumbent.
2. **Source-direction dependence.** A0/B1 all-target results beat the original
   controller by1.36044%-1.64708% across three seeds. A1/B2 all-target results lose
   0.61010%-0.90074% across three seeds. These directions cannot be selected from
   readout. More B windows do not, by themselves, establish better generalization.
3. **Learning objective versus downstream effect.**30/36 matched risk heads and
   33/36 OOF risk heads reduce fixed training-batch loss. Neither statistic proves
   superiority on new sources; six matched heads do not even reduce that fixed
   training loss. Every checkpoint is the preregistered final one, not a selected
   minimum-loss checkpoint. Utility losses use changing batches.
4. **Uncalibrated selected risk.**122/144 matched locality/views underpredict
   positive harm;66 exceed the nominal2% ratio. Worst realized selected ratio is
   0.15174 despite predicted ratios of0.00277-0.01272. This is not a certificate,
   even though net easy degradation passes in the opened development views.
5. **Unsupported extrapolation.**Every readout has only four independent source
   localities. Overlapping windows and repeated roles do not expand independent
   sample size. Unknown-label rows can receive predictions/interventions, but no
   accuracy or safety outcome is available for them. No cross-dataset or unseen
   confirmation result has been produced by this experiment.

## Next Falsifiable Repair

Keep A's original stopping-protected policy fixed on B and C. Train a matched
controller on B to predict incremental value/harm of overriding that policy,
with the old decision as the default. Separate added-switch and removed-switch
actions in the supervision so useful incumbent interventions are not discarded
by a floor-relative objective. Retain the same source exclusion, common event
cutoffs, seeds, simple control and pre-readout freeze. Compare against this round
and the unchanged incumbent on all registered development directions.

This next experiment is **not_run** here. Do not infer it works from the accounting.
It requires a new registration and genuine fitting. Further increasing model size,
opening confirmation sources, or tuning thresholds on these negative views is not
justified by the current result. Independent risk calibration and scene-joint
intervention evidence are still needed for the broader paper claim.
