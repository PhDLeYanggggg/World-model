# Risk-Conditioned Residual: No Promotion

## Result

Adding the frozen model's own causal risk scores does not establish a stable
improvement over the original risk estimator. The registered mechanism,
primary comparison and tail/coverage gates all fail. Deployment is unchanged.
This is a completed negative source-development experiment, not independent
confirmation, trajectory improvement or submission-ready evidence.

Full-input OOF score-plus-context correction versus the original estimator:

| Source assignment | Easy-harm MSE improvement (%) | 95% paired locality bootstrap CI |
|---|---:|---|
| Producer 0 / controller 1 | -0.2144 | [-1.2025, 0.4367] |
| Producer 0 / controller 2 | -2.2160 | [-5.5781, 0.0501] |
| Producer 1 / controller 0 | -1.6986 | [-5.6106, 0.6560] |
| Producer 1 / controller 2 | -1.0660 | [-3.6940, 0.4609] |
| Producer 2 / controller 0 | -0.3022 | [-3.0247, 2.3978] |
| Producer 2 / controller 1 | -0.1936 | [-0.7690, 0.2593] |

All six point estimates are negative and all intervals overlap zero. This is
neither stable improvement nor a proof of equivalence/safety. Three seeds are
averaged within each locality before 3,000 paired resamples of four localities
per assignment. Assignments overlap and intervals are not multiplicity-adjusted.

Against the previous common-event seven-feature correction, full-input points
range -0.2467% to +2.0981%, with all six intervals overlapping zero. Against
the prior three-locality in-sample context control, all points remain negative;
three intervals are below zero. Against the original inner-event OOF control,
two intervals are negative and four overlap. None passes the fixed criterion.

There are limited secondary signals, retained without promotion. Motion-only
versus common-event context has three positive intervals, but versus original
only one is positive and five overlap, with points -14.2686% to +1.6111%.
Matched cyclic controls also have isolated gains. They do not justify picking
a favorable assignment, input subset or producer after seeing its outcome.
All variants are shown in [results](results.md) and the
[comparison figure](risk_conditioned_residual.svg).

## What Was Run

- fresh_run: 144 original frozen-Torch fitting-row inferences; 432 exact
  producer-difference projections; 864 fixed closed-form probes over 144 views;
  36 source-held readout groups and 1,728 direct MSE arithmetic checks.
- cached_verified: 432 inner Torch heads, original outer estimators, frozen
  trajectory producers, historical control predictions and registered inputs.
- not_run: new neural/trajectory training, policy deployment evaluation,
  independent selection/calibration/confirmation, Stage5C and SMC.

All views had nonconstant score features and positive easy-harm fitting support.
Minimum positive fitting rows were 245 for full inputs and 24 for motion-only.
Maximum fixed-design projection error was 1.0613e-14. These are numerical
support and algebra checks, not statistical power or causal attribution.

Registration 4646c395 preceded support, support 1fb49d0d preceded fitting,
and prediction freeze c33fa80c preceded the new source-held readout. Summed
per-view fit/inference time was 57.873162 seconds; source loading, preflight
and verification are excluded. No new neural updates occurred.

## Interpretation

The two new scores help some weaker comparisons, but do not repair the strong
baseline comparison. This rules out these two fixed score-bin additions as
a sufficient repair. It does not prove that all model-aware calibration fails,
that OOF learning is useless, or that more epochs/larger models would help.

Only predicted H_E changes, clipped to the original H. Other costs and all
trajectories remain identical. Risk-estimation MSE is not trajectory ADE/FDE.
The next work must distinguish this fixed output constraint from limited
transportable conditional signal before another model change, preserving
all strong controls and independent roles. No held-selected shrinkage sweep.

Main protocol: observed 8 / predicted 12 native annotation steps, detector
image pixels. No metric, seconds, physical-safety, human-gold, true-3D,
foundation or publication-success claim. The research goal remains active.
