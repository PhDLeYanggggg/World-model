# Decision-Aligned Utility and Selected Risk

## Material Passport

Experiment execution protocol. Inputs: frozen source TRAIN packets, causal
eligibility verification and the failed completed easy-harm loss comparison.
Output: three matched cost-head arms, fixed-final development evidence and
unchanged risk gates. This is exposed development, not independent confirmation.

## Hypothesis and Prior Evidence

The completed deviance repair still underpredicts selected easy harm in72/72
views and fails both the stronger quadratic utility comparison and risk checks.
Test whether optimizing a smooth version of the actual intervention decision,
with empirical risk constraints, repairs selected utility beyond global moment
fitting. The hypothesis can fail; lower training loss is not its success criterion.

Task-aligned training is established prior art, not claimed as our invention.
Smart Predict-then-Optimize uses decision regret and a surrogate for its known
optimization formulation. Our learned constraint coefficients and missing
outcomes differ; this experiment is not an exact SPO+ implementation or a
reproduction of its guarantees. [Elmachtoub and Grigas, version5](https://arxiv.org/abs/1710.08005v5).
Task-based learning also optimizes downstream objectives through decision
problems. The present smooth independent gate is a task-matched baseline, not
their exact stochastic-program solver. [Donti et al., NeurIPS2017](https://proceedings.neurips.cc/paper/2017/file/3fc2c60b5782f641f76bcefc39fb2392-Paper.pdf).

## Fixed Data and Controls

Reuse all24 whole-recording source TRAIN packets and72 source/head identities.
Observation8/prediction12 with stride12, dataset-local coordinates and existing
partial-label masks remain unchanged. No seconds or metric conversion is made.
TRAIN-only normalization, known-query sampling, initialization, width32,
AdamW settings,2,000 updates and head seeds17/29/43 remain fixed. Forecast
producers are not retrained. Independent selection, calibration and confirmation
roles remain closed. History/scene features and deployment eligibility do not change.

Three arms: quadratic; quadratic plus task utility; quadratic plus task utility
and empirical all/easy-risk duals. This is216 fresh matched fits, not historical
quadratic checkpoint reuse. The new quadratic trainer must exactly reproduce
the original no-auxiliary trainer in the same environment during the pilot.
Historical cross-platform replay is not asserted. All arms use the same draws;
candidate duals and random/optimizer states are checkpointed and resume-tested.

## Objective

Let predicted moments be Bhat,Hhat,Rhat,ERhat,EHhat in frozen TRAIN scale units.
Let z=(Bhat-Hhat, .02Rhat-Hhat, .02ERhat-EHhat), with each coordinate divided
by its corresponding frozen TRAIN signed-score RMS. The training soft action is

```
a = causal_moving * TRAIN_support * product_j sigmoid(z_j / T).
regret = (1-a)*B + a*H.
c_all = a*(H-.02R); c_easy = a*(EH-.02ER).
```

All terms average agents within each sampled query, then queries equally.
Quadratic keeps the original objective exactly. Utility adds regret with weight1.
Risk adds regret + lambda_all*c_all + lambda_easy*c_easy with weight1. Both duals
start0 and update after each optimizer step by projected ascent with rate1,
clipped to[0,100], from the same pre-update minibatch violations. Temperature
decreases linearly from1 to0.1 over the fixed2,000-step budget. These are fixed
engineering choices, not validation-selected hyperparameters. The unchanged
hard deployment gate, not the soft gate, determines readout actions.

The risk loss is a TRAIN empirical constraint surrogate, not a risk certificate.
Multipliers reaching their cap, vanishing gates, empty selections, soft/hard
disagreement and harm underprediction must be reported. Empty selections cannot
pass undefined risk or positive-utility requirements. The2% definition is positive
selected harm/reference error, not net easy ADE degradation.

Wholly unknown future labels are excluded from TRAIN sampling exactly as in the
matched controls, not imputed zero/safe. This leaves missing-outcome uncertainty
unresolved; the unchanged completion bounds retain it in development evaluation.
True future/easy labels enter loss only. They never enter the soft action or
inference eligibility. No future endpoint, central velocity or test-endpoint goal
is an input. Causal movement is verified against the prior24-packet lineage check.

## Pilot and Execution Gates

Before training, run scalar loss/gradient/mask tests and exact interrupted-resume
tests locally using native arm64 Python,4compute threads and0DataLoader workers.
Then execute the first registered real TRAIN identity for100updates per arm,
plus a direct task-risk replay and original-control replay:500updates total.
Verify identical resumed/direct states, original quadratic equivalence, shared
draws, finite gradients/losses, storage reserve and projected full-run time.
This pilot is engineering evidence, not a model-quality promotion criterion.
It reads no validation labels. If it fails, retain the failure and repair execution
with an explicit amendment before retrying; do not silently change loss weights.

Source packets live on CREATE; local copies are receipts, not data. Real training
therefore runs through Slurm on the existing approved CPU runtime,4CPUs,8GiB,
zero DataLoader workers. No GPU or larger model is needed for width32 heads.
Reserve10GiB and cap new compressed checkpoints at256MiB. Use one owned job per
phase, no requeue, explicit resume and a12-hour wall bound; retain atomic snapshots.
Observe actual controller state and0:0exit for new jobs; historical database
availability is not fabricated. Preflight checks the current queue and ownership.

## Fixed Readout and Advancement

Freeze all216 finals before new development predictions. Retain original,
additive, Poisson and cost forest controls, contemporary quadratic and utility
arms, and the task-risk candidate. Historical deviance results remain a separate
development reference, not a substitute for contemporary matched controls.

Primary contrasts are candidate minus quadratic, utility and original, each
full-policy and same-query minimum-count paired lower utility. Require positive
nominal95%paired-locality lower intervals for all six primary contrasts, plus
complete finite support and no violation of the unchanged2%selected easy-risk
budget for candidate full and its three primary matched policies. Do not drop
unknown outcomes or unsupported localities. Original-selected score contrasts
remain undefined if any required support is missing.

Report all signed-score MSE, benefit/harm, positive easy risk, completion bounds,
intervention counts, per-locality results and tails. Bootstrap3,000paired
localities after averaging repeated views within locality. Three head seeds
are not independent forecast seeds. This protocol does not upgrade exposed
development into independent confirmation. No new thresholds or checkpoints
are selected from this readout. Failed development leaves deployment unchanged.

Passing permits design of the reserved independent study; it does not establish
CVPR readiness, physical safety, true3D or a foundation model. Stage5C and SMC
stay off. Source data, checkpoints and caches stay out of Git.
