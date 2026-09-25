# Symmetric Event-Risk Regression, Frozen Utility

Registered before any new fitting/readout. Source-development ablation only.

The preceding symmetric-utility experiment reduced utility bias but failed to
establish neural superiority over protected damping. Verified prior risk-head
diagnostics show neural easy-event predicted harm means 0.4610/0.4228/0.3651
versus true means 0.1782/0.1776/0.1832 pixels in seeds17/29/43. Predicted event
mass is 0.5030/0.4960/0.4895 versus true0.5400. These equal-locality population
means support a bias hypothesis; they do not certify selected-row risk or prove
that more neural intervention will be safe. Damping is also refitted fairly.

## Fixed Experiment

Change only neural event-risk loss from underharm4 to symmetric MSE. Retain
frozen symmetric utility heads, trajectories, 355 causal features, targets,
fitting-only preprocessing, seeds, sampling, 2% predicted-risk budget, source
support guard, bootstrap and joint-control settings. Ridge risk heads/decisions
are unchanged cached_verified controls, not fresh fits. No threshold selection.

Two candidates (neural/damping097), two event targets (all/easy), three seeds
(17/29/43), three outer source folds: 36 new risk heads, 2,000 updates each,
72,000 updates total. Width64/batch256/lr0.0003/clip5; checkpoint/heartbeat200.
A 100-update real pilot resumes inside the first budget. Every fit must finish
before comparative outcome readout. No new forecaster, utility or JEPA training.

Predict event baseline-error mass and positive event harm, not probabilities.
For equal causal forecasts, force harm to zero while retaining the predicted
baseline-error mass. Unknown targets never enter supervised fitting. Replay
every new head on4,096 rows and verify old/new preprocessing, constants, draw
counts and sampler RNG exactly. Compare against the matching frozen old head.

## Readout and Failure Criteria

Keep all48 policy views:24 newly scored neural-risk policies and24 unchanged
ridge controls. Fresh metrics on cached decisions are not fresh optimization.
Retain all five joint controls. Report all/hard/easy/zero-CV and supported FDE,
tails, worst locality, intervention counts, paired new-vs-old and neural-vs-
protected-damping contrasts. Keep undefined and failed-solver results visible.
Add fixed per-locality predicted/realized event mass and harm for all and selected
ADE-supported rows; zero mass gives an undefined ratio, not zero risk.
These are reliability diagnostics, not threshold calibration or tuned bins.

Observed safety remains worst-locality positive-easy degradation<=2% and zero
added error on observed zero-CV rows. Any additional accuracy accompanied by
violations does not pass safety. The prediction budget staying2% does not prove
the actual risk stays2%. No post-readout seed/view/checkpoint winner selection.

All three seeds,3,000 conditional source-locality bootstrap resamples. Twelve
opened source localities only. Reserved selection/calibration/confirmation roles
remain closed. Unknown-label inference rows retained. Released detector tracks,
image pixels,obs8/pred12 raw stride12: not t50,seconds,metric,human gold,physical
safety,true3D,foundation,independent confirmation or deployment certification.
No Stage5C/SMC execution and no deployment promotion.
