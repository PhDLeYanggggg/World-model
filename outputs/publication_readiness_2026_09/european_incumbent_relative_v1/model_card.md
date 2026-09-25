# Incumbent-Relative Intervention Model Card

## Intended Use
Research-only controller comparison on opened source-locality development data.
This is not a new trajectory world model or deployment replacement. Forecasts,
protected motion floor and original stopping controller remain frozen.

## Architecture And Supervision
Two381-input, width64, one-hidden-layer cost heads per arm/group. The utility head
uses a bounded simplex for expected positive benefit and harm. The risk head
factorizes reference cost, harmful-event probability and conditional severity,
with a fixed-scale cross-moment ranking auxiliary. Both use causal rollout
distance to bound absolute gain/harm. A learned risk estimate is not an upper
confidence bound and does not provide a certified improvement guarantee.

On rows where the incumbent chooses the floor, the alternate is neural; where it
chooses neural, the alternate is the floor. Incremental utility is positive/negative
parts of incumbent error minus alternate error. Incremental risk is incumbent
error and positive alternate harm on the registered all/easy event. Coordinates
and errors from future labels are used only for supervised costs or readout.

AdamW lr0.0003, weight_decay0.0001, batch256, gradient clip5,2,000 updates,
seeds17/29/43. Feature means/stds, CV loss scale and initialization use B only.
Source-balanced supervision excludes rows with entirely missing future labels.
The A-derived common easy/hard definitions are not recomputed on B or C.

Inference chooses an override only when the latest observed step is moving,
expected benefit exceeds harm, and predicted positive harm is at most2% of the
predicted reference cost. Otherwise the original action is preserved. The
floor-reference arm sees identical features and uses the same score test but
falls back to the floor. Add-only/remove-only are diagnostic restrictions on the
same incremental scores, not separately selected thresholds. Ridge alpha0.01 is
a fixed-capacity control, not a searched hyperparameter.

## Limitations
Only12 opened localities, four per readout and repeated across rotations.
Detector-derived tracks, partial future supervision and unstable low-error
ratios remain. No independent calibration or confirmation, no robustness
certificate, no proof of new scene/goal/interaction reasoning. Same-predictor
comparisons isolate a controller design decision, not increased dynamics capacity.
The threshold is unchanged, not validation-tuned in this experiment. Final-step
checkpoints are fixed by the preregistered budget, not readout-selected best.

## Forbidden Interpretations
No meters/seconds from image pixels/raw annotations, no human gold, true3D,
foundation, physical safety or submission-readiness statement. No Stage5C,
stochastic rollout, SMC or deployment promotion.
