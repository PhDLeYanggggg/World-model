# Matched Fixed-Floor Objective Ablation

## Material Passport

Fresh source-development experiment with frozen, provenance-verified
forecasters and controls. Not an independently confirmed main result.

## Research Question and Design

When an accurate causal fallback already handles easy trajectories, can
direct supervision of the positive-error budget improve neural intervention
decisions compared with separate cost moments? We freeze a protected damping
floor and neural prediction, and supervise Z=H-0.02R, where H is positive
incremental ADE and R is floor ADE. All/easy versions share the same past-only
inputs and matched architecture with the separate-moment MSE control.

The experiment keeps source4/4/2/2 exclusions, initialization, data draws,
preprocessing, utility, support guard and update budget fixed.108 new heads
are matched to108 verified cached controls. Decisions are frozen before
readout. A same-recording/frame equal-count control separates ranking changes
from abstention; no future frame can supply its intervention budget.

## Result Table

| Policy | ADE gain over floor % | Intervention % | Risk-budget failures /216 views |
|---|---:|---:|---:|
| Moment MSE |0.1544 [0.1160,0.1968]|7.8187|110;0undefined|
| Direct signed excess |0.1848 [0.1101,0.2753]|6.7818|79;14undefined|
| MSE at direct-model counts |0.2018 [0.1391,0.2755]|6.7818|99;14undefined|
| Frozen ridge |0.5233 [0.3833,0.6913]|5.6345|171;0undefined|

Direct supervision has no clear equal-count ADE advantage (-0.0168%, interval
[-0.0539%,0.0169%]); its fixed-roster harm primary is undefined. Fitting signed
MSE improves, but held signed MSE does not show improvement. These findings
reject this registered objective repair as a reliable safety solution.

## Interpretation and Limitations

The objective identifies only the signed combination, not individual
calibrated moments. Conditional squared-loss regression alone does not provide
valid upper-risk confidence bounds or simultaneous source guarantees. Easy
net preservation cannot replace positive-harm control. Three forecaster seeds
and3,000 locality-bootstrap draws describe12opened development sources;
overlapping windows and repeated fitted views are not independent units.

This is an informative negative ablation, not a novel world-model contribution
by itself. A viable main claim still requires nontrivial useful neural gain at
the prescribed risk budget, public strong-method comparisons, appropriate
scene-joint controls, source-independent calibration/confirmation and adequate
reproducibility. Image-local raw-frame detector-silver data cannot establish
metric, seconds-level, physical-safety, human-gold, true3D or foundation claims.
Stage5C and SMC remain disabled. See policy_and_positioning.md for primary
literature boundaries and the current official-date check.
