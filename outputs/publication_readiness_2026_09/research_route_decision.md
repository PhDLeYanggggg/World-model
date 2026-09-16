# Research route and 8-to-12 development decision

Decision date: 2026-09-16. Evidence status: protocol decision, not an experimental result.

The user selected eight observed steps and twelve predicted steps as the primary
task, with raw-frame t+50 supplemental, and delegated the research route. The
remaining choices below are researcher decisions made under that delegation;
they are not separately user-approved scientific findings.

## First paper: baseline-relative joint intervention

Working question: when does a neural trajectory forecast help more than a strong
causal forecast, and when does independently switching several interacting agents
produce an inferior joint decision?

The proposed contribution is cost-aware, scene-level intervention with separate
benefit/harm supervision. JEPA plus Transformer is not, by itself, a novelty claim.
The decisive experiment must compare the same predictor, out-of-fold supervision,
and actual intervention count against independent routing and a cost-sensitive
deferral control. It must test at least two credible forecasting backbones and
retain negative results. Predicted risk budgets are not statistical guarantees.

Target: CVPR 2027, conditional on convincing evidence, not promised acceptance.
The official call currently gives November 10 registration, November 16 full
submission and November 23 supplement deadlines (2026, AoE):
https://cvpr.thecvf.com/Conferences/2027/CallForPapers
It prohibits substantially similar concurrent conference/workshop submissions
during review. The separate 2027 Author Guidelines link is currently unavailable;
submission-time requirements must be checked again before any submission.

## Immediate real-data development study

- Primary task: eight consecutive observed annotation steps to twelve future
  annotation steps on each recording's native grid. No resampling, seconds or
  metric claim. This is not yet the standard calibrated ETH/UCY benchmark.
- Fit: ETH ETH, ETH Hotel, and UCY Zara01/02/03. Three physical-scene crossfit
  folds; all Zara recordings stay together.
- Development: UCY Students01/03, one physical scene. This is a deliberately
  limited transfer-development slice, not independent confirmation.
- Excluded here: oblique UCY Arxiepiskopi and repackaged PETS; neither becomes a
  hidden test set. Pending CITR/DUT source admission is not overridden.
- All included recordings retain `development_exposed` historical status.
- Reference floor: fixed causal constant velocity, selected by design before
  development evaluation, not called the strongest baseline without measurement.
- Primary error: past-normalized ADE with equal physical-scene weighting. FDE,
  native-coordinate per-recording errors, complete-label counts, easy/hard slices
  and intervention coverage accompany it. No pooled raw-coordinate error.
- Easy/hard thresholds: fit-only constant-velocity normalized ADE quartiles.
- Geometry proxy: fit Zara median nonzero displacement per observed step, with
  graph radius six times and proximity threshold half that value. This is a
  dataset-local diagnostic, not physical collision validation.
- Seeds fixed in advance: 17, 29, 43. A one-seed run is explicitly incomplete
  multi-seed evidence. Two thousand physical-scene bootstrap resamples are
  requested, but one development scene cannot establish a scene-level CI.
- Forecasters: fixed 1,000 updates, batch 128, width 64, two Transformer layers,
  learning rate 0.0003. Three leave-scene-out producers plus a full-fit producer
  per seed. No old Stage37 teacher or checkpoint enters this experiment.
- Heads: linear benefit/harm control and a small neural benefit/harm regressor
  trained on the identical verified out-of-fold rows. No hard winner labels.
- Two predeclared policy settings, not an open-ended threshold search. All arms
  are reported, including unrestricted neural predictions and fallback.
- Fixed-budget checkpoints, single-process loading, atomic resume and heartbeats.
- This study has no calibration/confirmation roles and cannot issue a formal risk
  guarantee, deployable claim or independent-generalization claim.

Raw-frame t+50 is a separate supplemental experiment after the main path is
validated. Exact native-grid availability must be reported; missing exact t+50
is not replaced by the nearest observation or relabeled as a seconds horizon.

## Evidence needed before submission

1. Finish the real-data mechanism comparison, cost-sensitive deferral and matched
   coverage controls. If joint decisions do not improve the matched independent
   control, remove the joint-intervention contribution claim.
2. Reproduce public strong forecasters under compatible sampling and geometry.
   Use original per-dataset units until calibration is independently verified.
3. Add genuinely untouched, legally admitted physical scenes. Split original
   recordings before generating overlapping windows. No renamed aliases across
   roles; no historical exposure reset.
4. Run all three seeds and scene-held-out development folds. Report uncertainty
   over independent scenes, not a narrow interval over overlapping windows.
5. Retrain no-interaction, no-scene, no-goal and cost-objective ablations where the
   modalities actually exist. Do not advertise absent modalities as implemented.
6. Freeze the selected method, then evaluate independent confirmation once.

## Later publication routes, contingent on distinct contributions

A journal extension aimed at TPAMI or IJCV would need substantive new evidence
and method: verified visual context, broader scene/domain coverage, robustness
under missing agents and sampling shift, and stronger generalization analysis.
An independent theory paper is justified only if a genuinely new result handles
the dependence and deployment setting; ordinary conformal calibration is not
automatically new. A dataset/benchmark paper requires sufficient new data,
provenance, permissions, adoption value and systematic evaluation, not merely
repackaging existing ETH/UCY/SDD files.

No parallel duplicate submissions, no arbitrary paper count target, no artificial
salami slicing. Prior papers must be disclosed and distinguished in any extension.
One strong, falsifiable contribution is the immediate priority.

## Claims retained throughout

M3W remains a 2.5D/pseudo-3D multi-agent trajectory research project. It is not a
true-3D or foundation world model. Historical Stage37/44 headline improvements
are not fresh independent evidence after the provenance audit. Stage5C execution
and SMC remain disabled. User confirmation is required before actual submission.
