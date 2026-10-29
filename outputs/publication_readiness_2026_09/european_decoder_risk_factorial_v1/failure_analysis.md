# Decoder and Risk Objective: What Failed and What Was Learned

## Complete Experiment, No Advancement

Fresh_run72 fixed-final cost-head fits and144 fit/held views are complete.
Three head seeds, six contexts, four sources/context and2,000updates/head were
fixed in advance. Eighteen exact old controls passed. Source-local heads and
forecast models were cached_verified. The experiment opened no outer development
or independent selection/calibration/confirmation roles. It is TRAIN-internal
recording-transfer research, not a new independently validated world model.

## 1. Source-Specific Output Capacity Is Not a Sufficient Repair

Without risk penalties, separate decoders reduce fit signed-score MSE by0.073962
versus a shared decoder (nominal95%CI[-0.107675,-0.044951]). On held recordings,
the difference is+0.256604[-0.096676,0.883664], with no supported improvement.
Held paired completion-lower utility is-1.762683pp[-2.519183,-0.977941], and
same-count utility is-0.580246pp[-1.100566,-0.057594]. Better fitting does not
transfer into better decisions. The495 additional parameters do not establish
source-semantic value or unseen-source generalization.

With risk penalties, separate decoding also fails: held MSE difference versus
shared risk is+5.709067[-1.488146,15.219810]. Full and same-count utility
intervals both cross zero. This is not proof that every source-specific design
must fail, but it rejects the registered fixed-budget repair.

## 2. Risk Penalties Trade Prediction Fit for Fewer Risk Violations

Within a shared decoder, adding risk penalties raises held signed MSE by5.756850
[3.125110,8.602720]. Within source decoders it raises MSE by11.209313
[3.046555,21.855650]. Fit MSE also rises in both comparisons. These are controlled
objective contrasts with identical draws/initialization, but not direct evidence
of conflicting gradient directions or a unique optimization mechanism.

Removing the penalties is not a safe fix. Held known easy-risk violations rise
from40 to61/72 with a shared decoder, and from42 to54/72 with source decoders.
Full completion support is1/0/1/3 of72 for shared-risk/shared-utility/source-risk/
source-utility. Every arm fails the complete support requirement.

The MSE difference-in-differences is+5.452463[-1.700635,15.012943] on held
recordings. Its interval crosses zero: a decoder-by-risk interaction is not
established. Pair-specific utility lower bounds cannot be subtracted to make
a corresponding interaction claim.

## 3. Smaller Prediction MSE Does Not Establish Useful Safe Selection

The source-utility model has lower held MSE than local controls:
-2.062072[-3.349375,-0.956808]. Yet its full paired lower utility is
-1.287145pp[-3.052476,0.123288], and its same-count result is
-0.568936pp[-1.078989,-0.109548]. Known easy-risk failures are54/72 versus46/72
for local controls. Its completion-upper easy violations are68/72, and61/72
views fail the net easy-preservation upper-bound check. It is not deployable.

The median predicted/observed selected easy harm is0.011443 in source-utility
versus0.069526 in source-risk (68 versus56 defined views). These are different
selected cohorts, so they diagnose underestimation but are not a controlled
calibration improvement comparison. Unknown selections also differ.

## 4. Apparent Safety Can Be Mostly Refusal, and Still Fail Transfer

Source-risk selects15,364 repeated fit occurrences versus67,774 for shared-risk,
with known easy violations1 versus9. On held recordings, it selects17,643 versus
27,826, yet has42 versus40 known easy violations. Eleven source-risk views select
nothing; twelve have undefined easy reference. Empty/undefined support is retained,
not silently scored as a safe successful intervention policy.

Localities007,110,119 and124 have6/6 known easy violations in every arm.
Other effects reverse by locality: source-risk worsens067 from1 to6/6 failures
but improves074 from5 to1/6 relative to shared-risk. No locality is removed.

## Evidence Limits

- Fit897,060 and held419,766 occurrences/arm repeat across views and head seeds.
- Missing outcomes18,978/9,228 remain in fit/held inference and conservative bounds.
- The3,000-draw bootstrap resamples12 localities after averaging repeated views.
  Shared training and prior exposure make intervals nominal, not independent
  confirmation or simultaneous confidence guarantees.
- Three cost-head seeds do not replace independent forecast-model seeds.
- No gradient-interference measurement, new trajectory model, domain zero-shot
  transfer, independent risk calibration or confirmation was run here.
- Raw-frame, dataset-local evidence remains non-metric and not seconds-level.

## Next Test, Not a Claimed Result

The next minimal repair should separate accurate gain/cost representation from
the mechanism controlling rare selected harm. Freeze a cost-fitting trunk and
measure a separately trained harm/tail branch on the same source TRAIN recording
partitions; use fixed-cohort diagnostics to distinguish cost distortion from
selection-induced underestimation. This is a proposed follow-up, not completed
training or an authorized opening of independent roles. Do not simply remove the
risk objective, add more source heads, tune held thresholds, or change the2% budget.

Keep all negative results, the source-local reference and exact controls. No
deployment promotion, Stage5C execution or SMC. The overall project remains
not yet submission-ready.
