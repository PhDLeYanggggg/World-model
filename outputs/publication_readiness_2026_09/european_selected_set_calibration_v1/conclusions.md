# Recalibrating the Retained Source Set Did Not Repair Risk

## Completed Experiment

The registered source-only control completed72 calibration groups and432
component/role comparisons on CREATE job37714473 (COMPLETED0:0,2m10s). Every
group was recomputed exactly. A separate scalar/inference implementation then
verified19,872 quantities,864 decision hashes and168 summary/bootstrap checks.
No new neural network or forest was trained. Forecasts and original calibrators
were cached_verified; selected-set calibration and readout are fresh_run.

The experiment retained the original2% selected-positive-harm/reference budget,
whole-recording OOF exclusion, three existing head seeds and all72 source heads.
Independent roles and transfer evaluation stayed closed. These are exposed
development localities, not independent confirmation.

## Primary Source-OOF Result

| Joint policy quantity | Raw-pool calibration | Selected-set recalibration |
|---|---:|---:|
| Complete finite-completion support /72 |23|23|
| Nonempty/defined easy-risk views |30|27|
| Undefined easy-risk views |42|45|
| Selected occurrences, dependent across views |76,483|67,160|
| Selected unknown occurrences |725|633|
| Easy-risk completion-upper violations |6/30|4/27|
| Worst completion upper ratio |20.0268%|28.0208%|
| Observed known-label easy-risk violations |2/30|1/27|

The complete pass count is unchanged: one view gained support and one lost it.
Selection retains87.8104% of the original occurrences. Fewer violations among a
smaller, changed population do not establish an improved risk controller.

Conservative utility change is **-0.00678063%** of full known reference-error
mass, nominal95% locality-bootstrap interval **[-0.01707456%,-0.00012607%]**.
This is the registered lower-utility accounting contrast, not an FDE improvement
or a changed risk denominator. All72 views contribute, averaged within12
localities before3000 resamples. The small effect and nominal, unadjusted
development interval should not be inflated into a broad accuracy claim.

On the27 common-defined easy-risk views, the mean completion-upper change is
+0.00872359 in ratio units, CI[-0.00031704,+0.02572623], across7 localities.
That interval includes zero; the strict all-view mean is undefined. It does not
prove a universal increase, but it certainly does not demonstrate the proposed
repair. The upper ratio includes unknown-label envelope mass and is not the
observed error rate.

## Component Controls and Resubstitution

Harm-only source-OOF decisions are unchanged:26/72 complete passes and zero
utility contrast. Reference-only gives24 rather than23 complete passes, but
loses conservative utility (-0.00543955%, CI[-0.01278803%,-0.00004845%]) and
reduces defined coverage from30 to28. This secondary arm cannot replace the
registered joint comparison after looking at its result.

Joint resubstitution removes the one completion-upper violation among its
defined views, but complete support falls26->18 and defined coverage30->23.
Its utility contrast is -0.00785210%, CI[-0.01825913%,-0.00031017%]. This is a
particularly clear example of why an in-source clean risk table can coexist
with worse useful coverage. Resubstitution is not held-out evidence.

## Decision

Reject this calibration control as a demonstrated repair. Do not run its
transfer evaluation, select a favorable arm/seed, increase iteration count on
the observed cases, or promote deployment. Joint OOF used at most6 rounds;
all modes used at most7 of the fixed8-round cap. There was no iteration-cap
exhaustion, so insufficient iteration is not the observed explanation.

The failure decomposition is in [failure_analysis.md](failure_analysis.md);
machine-readable evidence is in [verified_summary.json](verified_summary.json),
[joint_oof_details.json](joint_oof_details.json) and
[readout_verification.json](readout_verification.json).
The input packets and full coefficient/fold outputs remain on CREATE. Only
181,801bytes of light numeric evidence were collected, not raw arrays.

Protocol remains obs8/pred12, stride12 raw frames, image-local detector-silver.
No metric, seconds, human-gold, physical-safety, true3D, foundation or
submission-ready claim. Stage5C and SMC remain off.
