# Analytic Coupling of Frozen Binary Forecast Substitutions

**Question.** Does the actual interaction between two forecast substitutions
help relative-position prediction, beyond identical learned marginal costs?
This is a component test on existing exposed A donors, not a replacement
main metric, a new safety rule or an independent confirmation experiment.

**Fixed data.** Retain all 125,540 A donor rows, 14 sources, 34 recordings and
1,613 queries, observation 8 and prediction 12 at native stride 1. Use the
existing six source-excluded EqMotion seed-17 forecast banks, causal finite-
difference CV, original IDs and original masks. No old outcomes from own-B,
nested-C, H, DUT, reserved Moving20 or DroneCrowd are opened in this experiment.
No trajectory actor is retrained. Coordinates remain image-local.

**Model.** Edges are the undirected union of four nearest past positions within
each query. Relative position errors use a common edge scale: the geometric
mean of each agent's observed path length plus one pixel. The fit uses nine
vector predictors: current relative position, seven past relative increments
and the relative change between the frozen neural and CV paths at each step.
For each of twelve horizons, fit one rotation/reflection-equivariant linear
ridge predictor of the pair CV residual using shared coordinate coefficients.
Use observed common target steps as labels only; lambda is fixed at 0.001.
Each fitting source contributes its average over all queries, and each query
divides its supervised moments by requested edge-steps. Do not replace missing
labels by zero-error targets. Within each frozen actor design, fit on the
other donor source(s), never the held donor's labels. The original actor
also excludes all donors in that design. This yields 14 deterministic fits,
not 14 independent trials or three forecaster seeds.

**Mechanism.** Let e be the scaled baseline relative-position residual and
d_i, d_j the scaled path changes from CV to the neural forecast. For binary
switches a_i and a_j, the squared residual is
`||e + a_i d_i - a_j d_j||^2`. Its two marginal changes are
`2 e dot d_i + ||d_i||^2` and `-2 e dot d_j + ||d_j||^2`.
The mixed change is exactly `-2 d_i dot d_j`, independent of the future label
at that step. This elementary expansion is not claimed as a new theorem.
The predictions use the full requested grid and a fitted e, never future
support masks. Observed-label evaluation does not identify full-grid error.

**Matched comparison.** Freeze exactly min(2, number of query agents minus 1)
binary substitutions in both arms, before scoring. The additive arm minimizes
the same predicted marginal terms; the coupled arm also includes the analytic
mixed cost. Ties use ascending immutable row IDs. The small fixed count permits
exact enumeration without a timeout or tuning. It is a diagnostic intervention
budget, NOT the original policy's supported risk-constrained intervention count.
No claim that these actions meet the original all/easy 2% and easy-net limits.

**Readout.** Report available common-step squared relative-position error and
unsquared relative-position error with equal query weight within source;
retain all requested counts, zero-edge queries and missing support. Report
the existing agent ADE paired known and completion-lower contrasts separately,
over the unchanged roster, including easy/hard diagnostics. No finite squared
error completion interval is asserted for unbounded missing targets. Use 3,000
paired source bootstrap draws with seed 73107. If any source lacks pair support,
the pooled pair result is unavailable, not an average over selected sources.
Component encouragement requires positive squared and unsquared paired gains
with lower bootstrap endpoints above zero and no negative agent known gain.
The established whole-recording rules are unchanged and untested here. A pair
improvement alone neither licenses deployment nor substitutes for an ADE/risk
main result. No threshold, feature, loss or source selection after readout.

**Positioning.** Weng et al., [Joint Metrics Matter](https://arxiv.org/html/2305.06292),
define scene-joint forecasting evaluation, including JADE/JFDE and collision
statistics. Our relative vector error is not their joint sample metric.
Sun et al., [M2I](https://arxiv.org/abs/2202.11884), condition trajectories and
choose joint modes; here the trajectories themselves are frozen. Mao et al.,
[Regression with Multi-Expert Deferral](https://proceedings.mlr.press/v235/mao24d.html),
study label-dependent deferral costs; the extra factor investigated here is an
explicit interaction between different agents' forecast choices. The earlier
European contraction penalty reduced its own proxy but worsened observed pair
distance error. This experiment instead tests a labelled relative-vector cost,
not a physical collision surrogate or a novelty claim from a penalty alone.

**Execution.** Local arm64, NumPy/BLAS four threads, no worker process or GPU
probe. Save an atomic checkpoint per frozen design, with all scalar results
recomputed in a separate replay. No raw data, per-row choices, checkpoints or
third-party files committed. Stage5C/SMC off; no submission or deployment claim.
