# Anchored Causal Subset Supervision

## Role and Hypothesis

This is a new preregistered development repair, not a replacement primary for
the sealed query-excess experiment. Twelve already-opened development localities
and frozen 4 producer / 4 controller-floor / 2 risk-fitting / 2 held roles.
No independent selection, calibration or confirmation access. The hypothesis:
all-query training misses selected-subset error; retaining an individual-error
anchor and supervising fixed causal subsets may improve risk ordering.

## Matched Training

108 paired groups, 216 fresh 25,028-parameter heads, 2,000 updates each. The two
arms share initialization, fitting normalization, source-balanced query sampler,
known-row draws, features, architecture and optimizer settings. Previous frozen
pointwise/query controls remain cached_verified; new fits must match their
initialization and query/row draws as an additional check. No new forecaster.

Build three subset masks from all current causal rows BEFORE removing unknown
future labels: frozen controller admission among moving agents, lowest-envelope
half (ceil), and remaining highest-envelope half. Group by locality, recording
and current frame; deterministic row-ID tie break. The controller is fitted on
four sources disjoint from risk-fitting and held sources. Its original admission
is relative to CV, not necessarily admission relative to the protected floor;
it is a proxy subset, not a claim of exact deployed selections. The envelope is
a causal rollout-disagreement bound. No future coordinate, valid-future mask,
target latent, endpoint goal or central velocity enters subset construction.

For each all/easy signed error e, compare:

- subset_pointwise: 0.5 query-balanced mean individual squared error plus
  0.5 mean over three subsets of mean individual squared error.
- subset_aggregate: the same 0.5 individual anchor plus
  0.5 mean over three subsets of squared mean error.

Equal sampled-query and subset weights. Only known labels supervise; unknown
rows remain in causal banks/inference/action counts. Empty known subsets have
zero auxiliary loss in both arms; report coverage explicitly. Held subset MSE
is averaged over nonempty known subsets, unlike empty-inclusive training loss.
No threshold or weight search. Outputs remain score bases, not identified
expected cost components or calibrated probabilities.

## Evaluation and Abstention

Freeze training before actions and all actions before outcomes. Retain the
same 2% predicted all/easy budgets, fixed utility and 256-node solver bound.
For each new head evaluate independent, joint and risk-ranked policies. Rank
arms have exactly the original frozen independent action count per query.
Both inherit ten zero-action views; the old selected-risk ratio remains
undefined there, and the old primary remains incomplete. No roster deletion,
zero imputation or retrospective primary substitution.

Add a diagnostic with a fixed denominator: selected positive harm divided by
the total known protected-floor error, including nonselected agents. Verify
the denominator is positive in every view before fitting. Zero interventions
then give zero total harm while selected-risk is still undefined. This is NOT
the original 2% risk certificate; do not apply that threshold to the new ratio.
Always report coverage, action count and the original selected-risk ratio.

Four predeclared development contrasts: aggregate versus pointwise subset rank;
aggregate rank versus cached pointwise rank; aggregate versus pointwise subset
joint; aggregate joint versus cached pointwise joint. Report ADE gain, selected
harm reduction (nullable), fixed-denominator harm reduction, intervention-rate
difference. Rank contrasts are count-matched; joint contrasts are not. Existing
ADE/FDE, easy/hard, tail, partial/complete label and zero-CV harm metrics remain.
Bootstrap 3,000 locality-paired draws, seed101531, after averaging dependent views
within each of twelve localities; three training seeds17/29/43. These are
development uncertainty estimates, not independent confirmation.

The exploratory screen requires positive lower ADE and fixed-denominator harm
contrast bounds, count matching, joint ADE advantage, all216 joint views with
defined original selected-risk <=2%, easy degradation <=2% and no zero-CV harm.
Passing a diagnostic alone cannot promote deployment or certify risk.

## Compute and Verification

Native arm64 CPU4/interop1, workers0. A paired100-update pilot is included in
resumed training; atomic compressed model/optimizer/RNG checkpoints every500
updates, heartbeat and resume. Retain10GiB disk reserve. No new large feature
or latent bank. Reproduce first complete paired fit, all predictions/actions
and full evaluation. Independently check accounting, controls and query counts.
Record actual runtime, peak memory and unresolved checks separately.

Obs8/pred12 raw-frame stride12, image-local detector silver. No metric, seconds,
human-gold, physical-safety, true3D or foundation claim. No Stage5C, SMC or
deployment change. Signed-risk learning on fixed feasible subsets is related to
decision-focused solution-value supervision, not novel merely because it is
aggregated: Mandi et al., ICML2022,
https://proceedings.mlr.press/v162/mandi22a.html. This experiment is not a full
reproduction of their solver/ranking methods and supplies no new guarantee.
