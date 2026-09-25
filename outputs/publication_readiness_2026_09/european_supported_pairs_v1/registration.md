# Supported-Event Pairing: Controlled Training Registration

## Question and Prior Evidence

The preceding ranking-auxiliary study completed 36 new Torch heads and 216 views
(commit aa811b10). It did not establish safe neural superiority over equally
protected damping. Its neural/easy objective received 167,291 valid rank-pair
draws across 18,000 updates, versus 2,355,302 for neural/all. Pairing rows before
filtering undefined event mass can discard otherwise useful comparisons.

Hypothesis: pairing supported event rows first improves risk ordering at the
same intervention counts, without changing fitting examples, model capacity,
forecast quality or risk tolerance. This is a falsifiable optimization repair,
not a new architecture or risk guarantee. It need not solve the separate
realized-ratio versus conditional-moment estimand mismatch.

## The Only Learning Change

Within each original minibatch, define supported rows by fitting label B+H>0,
where B is reference error and H is positive harm. Keep their original order.
Within each locality, form cyclic adjacent pairs among those supported rows.
Skip equal-risk pairs as before. Singleton localities contribute no pair.
Zero-reference positive-harm rows remain supported. NaN, infinite or negative
labels cannot be hidden by filtering: reject them before constructing pairs.

The ranking logistic formula, absolute observed-risk-margin weights, normalized
pair-weight denominator, coefficient 1 and training-unit log epsilon 1e-6 stay
unchanged. Undefined event-mass rows remain in the original moment, occurrence
and severity losses. The same locality-balanced minibatch sequence is used.
There is no oversampling, additional random draw, extra training step, coefficient
sweep or outcome-selected arm. Pair exposure changes as the intended treatment.

Use the same initialization, 355 causal features, fitting-only scaler, 22,979
parameters, optimizer and 2,000 updates as the preceding ranked heads. Start
from matched initialization, not from extra-trained checkpoints. Old risk heads,
utility heads, trajectory forecasts and their full producer lineage are reused
with hashes. Both neural and damping candidates receive the treatment.

A fully-supported-label test must give identical losses and gradients. A
fully-supported fit and a zero-ranking-weight fit must match the old fit exactly.
Interrupted/resumed fitting must give identical predictions and diagnostics.
The frozen original implementation is not edited. The new training driver is
a separately identity-bound protocol snapshot because older hashes must remain
valid; shared numerical losses and evaluation helpers are imported unchanged.

## Fitting-Only Checks and Execution

Before fitting, measure pair support on 40 batches per head using only fitting
labels. Do not select an arm or coefficient from those statistics. Log the
original training losses and a fixed fitting batch at step zero and logged steps.
Its indices come from a cloned sampler RNG; this must not change optimization
draws. Fixed-batch losses are fitting diagnostics, not validation or convergence
proof. Preserve every trace, checkpoint, optimizer state and sampler state.

Run a 100-step actual Torch pilot, then resume all 36 heads to 2,000 updates:
three folds, seeds 17/29/43, two candidates and two event targets. Total 72,000
updates. Native arm64 CPU4/inter-op1/workers0, atomic checkpoints and heartbeat
every 200 steps. The prior whole fitting/replay/decision phase took about four
minutes, so local execution is proportionate. Disk below 10 GiB is a hard stop;
slow healthy execution is not a reason to reduce the matrix. No remote job is
submitted or altered by this experiment.

## Frozen Readout

Before any new outcome evaluation, replay every checkpoint and freeze all 36
causal decision banks. Preserve six views per group: full old/new policies,
both common-pool anchors, new order at old counts, old order at new counts.
Retain every one of 216 views, including support differences and harmful arms.
All 36 original-control metrics must exactly match the preceding ranked study.

Report all/easy/hard ADE, FDE, tails, worst locality, zero-reference harm,
intervention rate, predicted-budget violations, both ordering/count components
and direct neural versus equally protected damping. Use the unchanged CV
denominator and 3,000 paired locality-bootstrap draws. Separate scalar sorting
and coordinate arithmetic must verify decisions, metrics and decomposition.
Shared helper aliases product/hurdle mean frozen ranked control/supported-pair
treatment; no product-MSE fitting arm is claimed. Count-forced arms can violate
predicted risk and are offline diagnostics, never deployable candidates.

Ordering success requires consistent fold/seed evidence at both count anchors,
not a favorable full-policy score or one interval. Neural success additionally
requires superiority over equally protected damping and easy/zero-reference
preservation. A null result remains a null result even if pair counts increase.
No threshold is selected on this readout. No new deployment is authorized by
this development study, regardless of outcome.

## Evidence Limits

Each fit uses four fitting and eight complete-chain-excluded localities. All
twelve European Squares localities are already opened development. Three seeds
and 3,000 conditional locality resamples cannot restore independent final-test
status. Views share data/models and intervals are not multiplicity corrected.
These are detector-track image pixels, obs8/pred12 rawstride12, not historical
t50, seconds, metric, human gold, physical safety, true 3D or foundation results.
Historical Stage37 is not recertified. Reserved calibration and confirmation
remain closed. Stage5C and SMC remain disabled. Submission readiness is false.
