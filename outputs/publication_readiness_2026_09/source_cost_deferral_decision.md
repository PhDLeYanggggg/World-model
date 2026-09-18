# Training-Only Baseline-Relative Deferral Repair

## Material Passport

Prospective registration before any new real-data fit. Six new continuation
branches will be `fresh_run`. Three mask-only cosine controls, source inputs
and parent checkpoints are `cached_verified` after hash/schema checks. Held
source, main selection, risk calibration and confirmation are `not_run` in this
experiment. The main scientific task, units and data-role assignments stay fixed.

The previous turn made concrete progress: the matched modality experiment
completed and found no held-source benefit. Its post-hoc oracle ceiling concerns
only its saved candidate paths, not the potential of a differently trained head.

## Question

Can an observed-input decision head learn useful relative-cost intervention and
exact baseline fallback while retaining candidate learning? Or does it collapse
to predicting the baseline everywhere? Both outcomes must be distinguished from
generalization, calibration and world-model success.

The frozen previous head outputs small nonzero offsets on almost every supported
query. Exact-zero target rows account for 75.51-80.75% of its positive held-source
harm. The proposed repair changes training and permits an exact-zero output;
it does not tune a threshold on previously exposed held labels.

## Fixed Data and Compute

Retain all 15,430 complete stationary-history training queries outside bookstore:
545 scoped agents, 29 recordings, four source sites, original SDD train40 only.
Use the existing train-only feature normalizer, loss scale and observed frame.
No new split, metric, label eligibility or sampling change. Offline supplied
annotation histories may be interpolated; no strict sensor-as-of claim.

Continue the same verified mask-only ADE parents at step 2,000. Seeds 17/29/43,
two fixed loss variants, 8,000 new updates per branch: 48,000 new updates total.
Batch 64, AdamW, clip norm 5, cosine 0.0003 to 0.000003, checkpoints every 200,
full-training snapshots at 2k/4k/6k/10k. Copy the original optimizer moments and
sampler/Torch RNG; append empty optimizer state only for the new gate parameters.
All variants preserve sampled rows exactly. Reuse all three matched dense-mask
cosine controls from the preceding run, with no new control training claimed.

Local arm64 CPU4/inter-op1/workers0 is appropriate: the previous matched 48k
updates took 36.70 minutes with about 2.6 GiB observed RSS, and 66 GiB disk is
currently free. A 100-update pilot belongs to this fixed budget. Save PID,
heartbeat, atomic checkpoints, optimizer and RNG state. CREATE has only a saved
access blocker; current remote assets/jobs are unknown, and no job is invented.

## Model and Losses

Keep the same 44,864-parameter bounded proposal network and observed inputs.
Append a 33-parameter linear score head on the shared 32-dimensional hidden
representation. At initialization the proposal exactly reproduces the parent,
and the zero score means deterministic fallback. The threshold remains zero.

Let `e` be proposal ADE, `b` be stationary-CV ADE, `s` the fixed positive
training-only mean CV ADE, `z(x)` the learned score and `p=sigmoid(z)`.

1. `expected_cost`: minimize `mean(((1-p)*b + p*e)/s)`.
2. `cost_supervised`: add `SmoothL1(z, stop_gradient((b-e)/s))` and
   `0.1*mean(e/s)` to the same objective. The extra proposal term keeps training
   signal even when the gate leans toward rejection.

Inference returns the proposal only when `z>0`; otherwise it returns the exact
stationary baseline. Unsupported observed contexts also remain exactly baseline.
No future label is passed to either head. Future errors are supervision only.
The expectation is a differentiable training objective, not executed stochastic
rollout. The sigmoid is not asserted to be a calibrated success probability.
Only the supervised score is trained toward a signed normalized gain target.

The second variant changes both cost supervision and proposal-gradient support.
This is a fixed package comparison, not an isolated causal attribution to either
term. No hyperparameter search, temperature search or best-milestone selection.
All endpoints and intermediate training snapshots are retained.

## Evaluation and Falsification

Use complete training rows only. Report raw proposal and deterministic gated ADE,
FDE, native pixel error, zero-target absolute harm, training-defined hard slice,
moving slice, tail error, intervention rate, gate score fit, and binary future
oracle diagnostic. Report expected-action risk separately from deterministic
path error. Average seed errors, not model forecasts. Seed ranges are descriptive,
not independent-scene confidence intervals.

All-baseline output is a failure to demonstrate predictive improvement, even if
its harm is exactly zero. A training-side positive signal requires positive gain
over CV plus improvement over the corresponding dense control, positive moving
or hard gain, and no increase in absolute zero-target harm relative to that
control. This only licenses discussion of a separately registered source check;
it does not pass the main research/deployment gates. The main 2% relative easy
criterion is not silently redefined: here zero-CV error makes that percentage
undefined, so this is an absolute-harm diagnostic, not a replacement gate.

Verify exact parent proposal reconstruction, dense-engine equivalence, gradient
direction, detached cost-target gradients, label-free input signature, matched
samplers, exact prediction replay and completed-resume immutability. These are
engineering checks, not proof of predictive information. No held forecasts are
allowed by this runner, regardless of training outcome.

## Prior Work and Novelty Boundary

Joint predictor/rejector optimization can itself fail; regression-with-rejection
research explicitly analyzes this difficulty. Our trajectory ADE and
sample-dependent baseline cost do not inherit its squared-loss assumptions or
consistency claims. [Li et al., AISTATS 2024, Sections 1-2](https://proceedings.mlr.press/v238/li24g/li24g.pdf).

Regression with expert deferral is an established framework, not a new M3W idea.
This small head is a repair/control, not the paper contribution. The outstanding
contribution still requires relative-risk learning, scene-level joint actions
and independent calibration evidence. [Mao et al., ICML 2024](https://proceedings.mlr.press/v235/mao24d.html).

Reducing intervention does not automatically protect every subgroup, motivating
explicit zero-target and hard-slice reporting rather than a coverage-only
success claim. Our slices are not that paper's fairness groups and do not inherit
its guarantees. [Shah et al., ICML 2022](https://proceedings.mlr.press/v162/shah22a.html).

No metric, seconds, human-gold, true3D, foundation or submission-ready claim.
Source task remains 8-to-12 at stride12/+144 raw frames. No deployment change,
Stage5C execution or SMC.
