# Bounded Repair: Train the Recording Decision Objective

## Material Passport

SOURCE development, 29 October 2026. Registered after the frozen multi-action
cost readout fails, before any new policy update. The old 54 fits, all144
readout groups and negative result remain immutable. No independent role is
opened, and no risk threshold or recording roster is relaxed.

## Evidence and Hypothesis

The five-moment cost rule uses the neural action in18,546 fitting interventions,
so failure is not simply all-fallback behavior. It nevertheless supports only
77/444 fitting recording/head views. There are303 easy-harm-upper violations
and36 undefined-reference views, with overlapping reasons. Thus accurate-enough
local moment regression plus a predicted2% check is not solving the actual
recording decision objective. The next hypothesis is that direct gain/harm
decision training with exact whole-record gradients can improve that mismatch.
It may still fail to generalize; no success is presupposed.

## Frozen Inputs and Matched Comparisons

Freeze all preceding cost heads and the original ADE-only neural producer.
Train a compact causal decision correction for each of the same six contexts,
three head seeds and three banks: physical-only, physical+neural, neural-only.
Fitting/inner-held partitions, no-op removal, native masks, floor, reference,
easy event and completion envelopes stay unchanged. No new raw data is added.

Each action has twelve causal values: five frozen predicted moments divided
by the frozen fitting-source cost scale, six existing motion/neighbor/rollout
descriptors, and maximum rollout envelope divided by that scale. Values are
clipped at fixed8; no new fitted or held normalization. Append the same seven
action identities. A width32 MLP outputs a logit correction, initialized to
zero around the exact preceding pointwise action. Floor logit is zero; analytic
floor copies are unavailable. Inference never reads outcomes or record labels.

## Exact Finite-Record Objective

Use a hard one-action-or-floor forward decision and categorical straight-through
gradients, with temperature1to0.1 over500fixed full-gradient updates. Benefit
minus harm is the utility; a wholly unknown selected action receives its
negative maximum-disagreement completion utility. Unknown rows are not zero
errors and remain available at inference.

Each fitting recording has separate dual constraints on
`sum(H - .02 R + unknown_envelope)` and
`sum(EH - .02 ER + unknown_envelope)`. Add a positive real easy-reference
support constraint equal to one minimum supported eligible easy-row reference
mass. This prevents treating an empty empirical ratio as a training success;
it does not add fake floor anchors or change the evaluation2%bound. Failure
of the neural-only bank to have safe support stays possible and is reported.

Use exact whole-record sums for gradients and dual updates, chunked only for
memory. There is no sampled risk estimate that can miss a rare easy row.
Source-balanced row weights preserve the fitting population weighting; fixed
record scales change units, not constraint signs. Dual learning rate1, cap100,
AdamW0.001, gradient clip5,500updates are fixed prospectively. All three banks
have matched head initialization and update count. This is a new decision head,
not a single-factor claim about the preceding regression network.

## Run and Evaluation

Native arm64 Torch CPU4/interop1, num_workers0, checkpoints50, heartbeat25.
Run a real5+5/direct10 exact-resume pilot on all three banks; estimate full
cost from it. All54 endpoints /27,000 full-gradient updates must be frozen
before readout. Keep10GiB free and up to12h per healthy process. Resume the
same roster after interruption. No downgrade for slow computation.

Evaluate all required fitting and exposed inner-held recording/head views.
Compare the preceding moment-threshold policy and its frozen physical-only
and binary controls; use the same full2%requirements, unknown completion
bounds and utility-only count matching. Reuse the paired locality readout,
three head seeds and3,000bootstrap draws. Report all final endpoints, not a
best seed/checkpoint. No model or cutoff is chosen on independent data.

This is recording-aware training of a causal local policy, not yet a scene-joint
interaction optimizer. A pass would still need the separate matched scene-joint,
published-deferral and independent calibration/confirmation experiments.
No metric,seconds,true3D,foundation,Stage5C orSMC claim or execution.
