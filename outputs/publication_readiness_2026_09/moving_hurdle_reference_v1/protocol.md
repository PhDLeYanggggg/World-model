# Source-excluded hurdle reference prediction

4 November 2026. The preceding fixed-decision audit finds47active shared-policy
recording views without any observed selected easy reference, despite every
selected row having some future label. Upward harm refiners freeze R/ER. This
study tests a missing reference-event target; it does not assume that another
harm penalty or generic three-seed uncertainty feature can repair it.

## Prediction task, not a new policy

For an agent's available-label easy-reference amount ER, use the standard
factorization E[ER|X] = P(ER>0|X) E[ER|ER>0,X]. Both terms condition on the same
original source/query weighted population. Conditioning positive labels does
NOT rebalance queries/sources a second time. This algebra is not a new
architecture or a calibrated record-level confidence guarantee.

Compare prior, direct64-tree regression and hurdle32-classification plus
32-positive-regression trees. Same depth6/min_leaf64/max_features32, no bootstrap,
single-process ExtraTrees. Equal64-tree budgets are not parameter-exact models;
actual nodes are reported. Targets are ER divided by the agent's causal scale,
not log-transformed means. Reconstruct physical reference by multiplying by
that same past-only scale. Unknown outcomes are masked out of supervised loss,
never assigned negative event labels. Retain their inference rows and counts.

382features: original raw local380 causal descriptors at the fixed full neural
proposal, log current query size and log causal history scale. They use past,
current context and producer-excluded predicted rollouts, not future truth.
No learned cost prediction/hidden state, original C-fitted normalizer, source
identity, future validity/easy label or test-endpoint goal is an input. Forest
thresholds learn only on the actual inner training rows. No new shared48input.

## Internal model selection

Keep original producer P / critic-fitting C / held H roles unchanged. The
producer excludes all C and H sources. Internally hash-sort C source names
with reference_inner_v1 and assign round-robin3folds, without outcomes. Fit
each head on C excluding the entire inner-held source groups; predict every
inner-held row. Source/query loss weights use observed eligible training rows
only. These are internal training-validation folds, not new official risk
calibration or final-confirmation roles. Neither the old C-trained critic nor
its normalizer is used as an inner input, avoiding upstream inner exposure.

Select among prior/direct/hurdle by complete inner-OOF normalized ER MSE, with
fixed prior/direct/hurdle order as tie break. Record MAE and event Brier/AUROC/
AUPRC/log loss separately. The prior is fitted inside each inner train fold.
Fit both candidate families on full C afterward for predeclared ablations;
retain the inner-selected choice. Complete all18jobs/216primary forests before
new H predictions. No H outcomes select models, cutoffs, weights or epochs.
All23supported SOURCE groups remain already-exposed development, not an
independent test. Original24primary remains not_estimable.

## H evaluation and limits

Predict all H rows from their causal packet before constructing evaluation
labels. Report prior/direct/hurdle/inner-selected normalized reference MSE/MAE,
event scores, per-source statistics and3000source-group bootstrap after
producer/seed averaging. All three real seeds are trained. Event estimand is
the supplied available-label ER>0, conditional on observed-label supervision,
not complete12step future labels or physical-world support. Missing-at-random
is not assumed. No independence of agents is assumed to construct a probability
that a record has any positive-reference event.

Keep every Stage37/current policy and trajectory unchanged. A future fixed-set
support audit may inspect new predictions, but no allocation or deployment
change is part of this first prediction study. A head MSE/Brier improvement
alone does not satisfy trajectory utility, original2%risk or submission gates.

## Execution

Commit code/config/protocol and registration before any real fit. Pilot all
3critic source folds,3tasks each:16direct versus8+8resumed trees,9exact cells,
288verification tree fits. Two full candidate families on the first job are
refit (3forests/128trees). Primary216forests contain9216trees; all verification
fits add416trees if uninterrupted. Checkpoint every8trees; complete files and
input hashes must reconstruct exactly on resume. Interrupted unfinished blocks
are not claimed preserved. Heartbeat and PID, arm64CPU4/interop1/workers0,
12hphase boundary,5GiB disk reserve,512MiB private cap. Local pilot measures
actual time/RSS before full placement. No default medium/quick substitution.

Reserved20Moving groups, DUT calibration and DroneCrowd confirmation remain
closed. No new images, raw data download, trajectory training, simulation,
metric/seconds/true3D/foundation/human-gold claim. Native8/12annotation steps and
available-label errors retain their limitations. Stage5C/SMC remain off.
