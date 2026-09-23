# Fixed OOF Gain/Harm Heads for the External Study

Registered before fitting or external readout, 2026-09-23. This prepares the cost
estimators for the six frozen source-only predictors. It is not a new positive
experiment, independent confirmation, novel regression method or calibrated policy.

## Data and Exclusion

Use the same 175,756 approved past-eligible SDD queries and four development-exposed
sites. For every training row, obtain its candidate trajectory from the existing
matching-family, matching-seed predictor trained on the other three sites. Check
its checkpoint train IDs, normalizers, loss factors and sample counts against that
exclusion. Verify the cached prediction hash and exact global row alignment. There
are 24 such predictors: four sites, three seeds and two families.

Do not use the new full-source predictors' in-sample errors as OOF labels. Their
checkpoints only identify the future deployment candidates and remain unchanged.
Do not claim that an in-sample risk-head fit is an OOF evaluation of the composite
policy. Existing pair-excluded producers remain the appropriate route when an
outer SDD policy evaluation itself must exclude another site.

Use complete twelve-step future supervision for cost fitting, as in the prior
bounded-cost experiment. Partial future observations remain in the indexed source
population but receive no invented full-grid cost labels and zero head-sampling
mass. This training-label restriction is never an inference-time agent filter.
The resulting complete-case selection bias is a limitation, not a missingness fix.

## Fixed Models and Objective

For each family and seed, fit the existing width-64 bounded continuous gain/harm
MLP, 3,000 updates of 256 rows, then a 128-tree ExtraTrees comparator with depth16,
minimum leaf64 and feature fraction1/3. Six neural heads plus six forests. Use the
existing learning rate, checkpoint cadence and strict arm, without search or early
stopping. The neural budget is 18,000 updates / 4,608,000 draws. Forests reuse the
exact neural per-row draw counts; these are not additional independent samples.

The two targets are positive and negative parts of native ADE gain over causal CV.
Divide both by the causal mean candidate/CV trajectory disagreement. By the reverse
triangle inequality their nonnegative sum is at most one. The neural head uses
the established bounded_fraction loss and parameterization. The forest regresses
the same two fractional targets with squared error and weights equal to neural
sampling counts. No extra distance weight or risk-dependent reweighting is used.
Distance-zero rows have exactly zero neural loss/gradient and zero forest weight.
This changes only an overall constant in the empirical objective, not its optimum.
It does not equate neural SGD with tree optimization or model capacity.

Use the established 356 causal features: past motion and neighbors, CV rollout,
candidate rollout, observed scale and prediction disagreement. Source-fitted
standardization is independent of external data. No future target, label mask,
scene identity or recorded best-baseline label is an inference feature. The two
predicted costs are not calibrated probabilities or statistical safety guarantees.

## Readout and Recovery

Keep all twelve endpoints; no source-loss winner or external-score winner is
selected. Fixed source-input probes check finite outputs and the cost bound after
all fits finish. Do not report their training error as generalization. A separate
replay process must reload all checkpoints and reproduce probe scores exactly.
Log training losses, forest fitting diagnostics, sampling coverage and compute.

Run locally with native arm64 Python, four compute threads, one Torch interop
thread, no loader multiprocessing. Forest fitting uses the threading backend.
Atomic neural checkpoints preserve optimizer and sampler RNG; forest checkpoints
preserve deterministic warm-start trees. Time a registered 16-tree prefix and
continue it in place. Move to CREATE only if measured cost justifies it; do not
shrink the registered budget because of slowness.

DUT remains calibration-reserved and DroneCrowd confirmation-reserved, with no
arrays or predictions opened by this run. The whole-source reservations and
source-use conditions remain unchanged. Independent support is still limited;
there is no metric, seconds, true3D, foundation, independent safety, Stage5C or SMC
claim. Freezing these heads is necessary but not sufficient: the complete joint
controller, supported external admission and one-shot readout rules remain to be
bound before external confirmation. Negative development evidence is retained.
