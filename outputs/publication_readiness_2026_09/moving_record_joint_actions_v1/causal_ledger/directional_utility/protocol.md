# Directional ADE Utility At Fixed Risk Costs

On exposed Moving A, predict the value of an existing candidate rather
than modifying its trajectory. Let e = target - constant-velocity
forecast, and d = candidate - constant-velocity forecast. The per-step
gain is ||e|| - ||e-d|| = dot(e/||e||, d) - R, where convexity gives
R >= 0. Use the zero subgradient at e = 0. The shared direction and
candidate-specific nonnegative remainder are supervised only on fitting
labels. Use per-step direction ridge fits in the observed heading frame
and candidate-specific remainder ridge fits (both lambda .002). Rows
with missing future steps contribute only observed labels. Predict
direction at all 12 steps at inference; never take the target mask as
input. At rest the local frame defaults to image axes.

Reuse the size study's seven fitting sources in each held-source cell.
The other seven, including the held source, contribute no fit statistics.
The original predicted all/easy costs, missingness envelope, fixed
candidate paths, and original per-query witness and count are identical.
Recompute the direct-gain control and require exact replay. Compare
direct gain, shared tangent only, and shared tangent minus learned
nonnegative remainder. No selection of thresholds, hyperparameters,
or continuation by held outcomes. Primary is directional minus direct
paired completion-lower utility, with 3,000 whole-source bootstrap
draws after averaging three forecaster seeds. Also report known-label
gain, exact count matching, changed choices, and unchanged 2% and
required-support pass counts.

Multi-expert regression deferral already exists (Mao et al.,
[ICML 2024](https://proceedings.mlr.press/v235/mao24d.html)). This is
a component test of structured gain for one trajectory bank and a fixed
recording allocator, not a reproduction of their method or a claim of
guaranteed risk. Keep 8 observed / 12 future annotation steps in image
coordinates. No external outcomes, deployment, Stage5C or SMC.
