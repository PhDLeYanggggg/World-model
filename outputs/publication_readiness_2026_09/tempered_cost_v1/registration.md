# Fixed Intermediate Disagreement Weighting

Motivation: the completed train-defined diagnostic finds high-disagreement
benefit underestimation and harm overestimation already inside fraction-head
fitting, amplified on deathCircle. Native-cost fitting ranks that stratum better
but the previously fixed native policy fails protection. This motivates one
fixed interpolation, not threshold tuning or a search over exponents.

Fit12newheads,4source folds x3seeds17/29/43, using exactly the frozen
EqMotion-specific training views, initialization,22,978parameter bounded head,
preprocessing, source-uniform draws and3,000update budget. The only training
change is loss weighting. With native costs divided by the fitting-only CV cost
scale and similarly scaled forecast disagreement d, use
`mean((predicted_cost - target_cost)^2 / d)` for d>0. At d=0 both costs and
predictions are exactly zero, and denominator1avoids undefined arithmetic.
Native and fraction controls have exponents0and2; new exponent1is fixed.
This numerical zero rule does not relax zero-reference evaluation protection.

Frozen controls are the complete predictor-specific direct/native/fraction
study, same candidate forecasts and rows. No retraining the forecaster, new
features, sampling curriculum, threshold adjustment or fitted calibration.
Retain net-stop and strict-stop(harm<=0.1benefit), plus matched-count ranking
using the OLD FROZEN transferred fraction-strict count. Fixed source readout
only; original closed roles remain closed.

Primary contrast: new strict minus refitted fraction-strict equal-site ADE gain.
Require positive lower paired3,000scene-bootstrap endpoint, positive CV gain
for every seed, every seed's easy degradation<=2%, and zero observed complete
exact-CV harm. Report all policies, seeds, scenes, FDE, hard/easy/tails, missing
outcomes and full-grid average-gain bounds regardless of pass/fail. A favorable
secondary arm cannot replace this primary. Constant core/threshold comparisons
are not independent deployment calibration.

Freeze decisions before outcome reductions; replay all12heads and verify
arithmetic separately. Check matched draw vectors against prior heads. Save
config/checkpoint/heartbeat and resume atomic checkpoints, CPU4/inter-op1/workers0.
The four sites are design-exposed. A loss interpolation is not automatically a
novel paper contribution; independent confirmation and method positioning remain
separate requirements even if this empirical test passes.

Eight observed/twelve predicted sampled annotation steps, SDDstride12, annotation
pixels. No verified metric/time, true3D, foundation, deployment, Stage5C or SMC.
