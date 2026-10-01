# Source-Validation Decision-Utility Selection Control

## Material Passport and Hypothesis

Registered after the negative source-checkpoint experiment, before any new
source-validation policy readout or transfer decisions. Does choosing a frozen
head using actual source-validation utility under the existing all/easy risk
screens work better than choosing it only by global signed-score MSE?

This is a developmental model-selection test, not independent calibration,
new neural dynamics training or a promised deployment improvement. Keep the
same 12 exposed localities, three seeds, 72 original source fits, 216 directional
views and 4/4/2/2 producer/controller/fitting/outer exclusions. Independent
selection/calibration/confirmation sources remain closed.

## Fixed Candidates and Rule

Reuse the parent's frozen initial training-prior head, final-step head and
MSE-selected head. No weights are updated and no new checkpoints are selected
from transfer outcomes. The earlier source-validation recording partition is
unchanged. It has already selected the MSE candidate, so the following safety
screen is explicitly a development-selection criterion, not a calibrated bound.

For each original training locality/context, evaluate all three frozen heads on
that source's validation recordings only. Compute causal actions before reading
their validation labels: positive predicted benefit minus harm, predicted all
and easy excess <=0, motion eligibility and optimization-source support limit.
Do not alter these per-row action rules.

A candidate is source-validation-supported only if all conditions hold:

1. No selected validation outcome is unknown.
2. Its selected all-reference and easy-reference denominators are both positive.
3. Actual selected positive harm / selected reference <=2%, separately all/easy.
4. Whole-population validation easy ADE degradation <=2%.
5. Validation net ADE gain over the frozen floor is strictly positive.

Among supported candidates choose greatest validation net gain. Exact ties
prefer fewer interventions, then fixed order MSE, final, initial. If none pass,
choose the frozen floor for this source head. Future label availability is used
only to assess a model on source-validation labels, never to gate an inference
row. Unknown transfer outcomes are retained and reported. No threshold grid.

An empty-intervention policy has undefined selected risk and is labelled
unsupported fallback, not a successful risk pass or learned improvement.
The initial head is a training-prior control, not learned representation lift.

## Freeze and Evaluation

Commit source-level choices and their validation evidence before generating
transfer decisions. Commit all 216 causal decision hashes before outcome readout.
Compare MSE selection versus this decision-aware selection, both independently
and at identical per-query intervention counts. An always-floor policy remains
an explicit reference. Count matching can change composition, not just ranking.

Primary developmental contrast: same-count ADE improvement over MSE-selected
policy. Also report all/easy selected risk, undefined references, unknown
interventions, worst easy ADE degradation, all/hard/FDE gains, coverage, all
locality means and seed-specific contrasts. Use 3,000 paired locality bootstrap
draws after averaging within each of the 12 localities. Intervals are nominal
and do not correct repeated adaptive development or shared producers.

If fallback removes all interventions, that is evidence the source-validation
rule could not support a neural policy, not success. No independent confirmation
is opened to rescue an unfavorable result. No deployment promotion from this
development-only experiment even if point estimates improve.

## Reproduction and Scope

Validate every parent source/artifact hash. Recompute source choices exactly;
recompute all transfer action hashes before evaluation; repeat the full readout.
Use separately implemented metric accounting. Native arm64 CPU4/interOp1,
workers0, process lock and heartbeat. Only small aggregate/hashes are written;
no new checkpoints, row caches, raw data exports or GPU jobs are needed.

Current coordinate/label boundary: image-local detector-silver, obs8/pred12 at
stride12 raw frames. No metric, seconds, human-gold, true3D, foundation or physical
safety claim. Stage5C execution and SMC remain off. This does not supersede the
full source/query-aware neural intervention research objective.
