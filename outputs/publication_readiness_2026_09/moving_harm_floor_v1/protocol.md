# Harm-Weight Allocation With An Equal-Mass Control

## Problem And Hypothesis

The preceding joint loss reduces false rejections but increases harmful
admissions. Fitting-only attribution finds 4/17/141 newly admitted harmful
proposals. Their median normalized excesses are below the corresponding
harmful-proposal medians. The total positive target mass already exceeds the
safe-benefit mass, refuting a simple benefit-mass-dominance explanation.

Hypothesis: absolute-target weighting undertrains lower-magnitude positive
excess decisions. A positive-only penalty floor may improve their rejection.
This does not establish novelty or guarantee safe generalization. A uniform
equal-total-positive-mass control distinguishes allocation from scale.

## Fixed Models And Loss

Nine real neural fits: three original own-B contexts, three arms:
joint, uniform_harm, harm_floor. All retain the separate64 architecture,
64,260 parameters, 492 causal inputs, identical original warm starts,
fitting-only preprocessing, source-excluded proposals and query draws.
Fresh joint controls must exactly reproduce the previous joint models,
optimizers, traces, sampler states and predictions.

Keep L0 = 0.5*risk_MSE + 0.5*decision_objective unchanged. Normalized targets
y, predictions r,d, and observed joint auxiliary loss are:

    p = mean_channels(max(y, 0))
    n = mean_channels(max(-y, 0))
    t = p if p > 0 else -n
    score = max(r_all, r_easy, d_all, d_easy)
    auxiliary = 4 * weight * softplus(-sign(t) * score)
    L = L0 + 0.5 * query_average(observed * auxiliary)

joint: weight = abs(t).
harm_floor: weight = max(p, m) for positive targets; otherwise abs(t).
uniform_harm: weight = c*p for positive targets; otherwise abs(t).

m is the own-B weighted median of positive p. Weights are the exact empirical
query weights from the prespecified complete 2,000-update sampling schedule,
not a sample-count estimate or held statistic. Unknown and unsupported rows
cannot determine m or c. Positive support must exist or fitting stops.

    c = sum_fit(w * max(p, m) * positive_mask)
        / sum_fit(w * p * positive_mask)

Thus the two treatment arms have equal positive coefficient mass under the
same query weighting. They need not have equal realized gradient or loss after
fitting; these are outcomes, not a claimed invariant. The median is computed
once per own-B context, stored with each checkpoint and frozen before readout.
It is a training weight, never an inference threshold or feature. Safe and
exactly zero targets, original risk MSE and unknown supervision are unchanged.
No test-selected scale, hyperparameter search, source-specific held tuning,
new goal, future input, central velocity or new latent is introduced.

## Budget, Selection And Readout

AdamW lr 0.0001, decay 0.0001, gradient clip 5, complete-query batch 72,
2,000 fixed updates per head, at least four visits per fitting query. Final
iterate only, not validation-selected. The first full harm_floor fit repeats
exactly. Every arm has a real 50+50 versus 100-update recovery/prediction pilot.

All nine fits freeze before additional exposed-development scoring. Keep the
existing internal expansion screen: all three harm_floor contexts reduce
MSE at least 1% from warm and do not worsen either channel sign cost, using
the same 1e-12 allowance. This is an internal diagnostic, not a safety gate.
If it fails, additional held readout is not_run; no alternate arm is selected
post hoc. All fitting outcomes, including negative controls, are reported.

If allowed, use only the same seven already exposed SOURCE units / 17 views,
one producer seed, nested C. Readout names are uniform_harm, harm_floor, joint;
primary paired contrast is harm_floor minus uniform_harm. Also report joint
control utility, constant control, risk-only diagnostics, count-matched NA,
worst easy-net, per-source utility and full original risk/support. Freeze all
predictions/actions before outcomes. Replay all caches and metrics exactly.
Missing-label completion bounds are not sampling CIs. Do not claim independent
confirmation, statistical significance or full-roster 12-step ADE.

## Unchanged Scientific Contract

Original 2% all/easy positive-harm, easy-net and support requirements stay.
No identity certification; empty controls/missing support remain NA. The
original 24-group primary remains not estimable, not passed. Reserved20
Moving, DUT, DroneCrowd and H stay closed. No deployment upgrade follows
from an internal training screen. Stage5C and SMC stay disabled.

This is neural intervention-risk learning, not new forecasting dynamics or
JEPA. Native obs8/pred12 image coordinates; not metric, seconds, true 3D,
foundation or human-gold evidence. Public artifacts exclude private arrays,
weights and third-party data. Local permitted assets are necessary for replay.

## Runtime And Reproduction

Native arm64 .venv-pytorch; CPU 4/inter-op 1/workers 0, no resource probing.
Measured preceding six-fit run: 1,051 s, peak 7.56 GB. Nine same-size fits
should be practical locally, with cost measured again rather than assumed.
Checkpoint/heartbeat every 100 completed steps; atomic checkpoint updates,
exact resume, 5 GiB disk reserve and 512 MiB new private-artifact cap.
Healthy runs up to 12 h are allowed. No new CREATE or duplicate job.

Run `.venv-pytorch/bin/python scripts/run_m3w_moving_harm_floor.py PHASE`
for register, pilot, train, evaluate, verify; interrupted phases use --resume.
Commit code, configuration and registration before training. Save complete
fitting reports and negative results; no success is inferred from execution.
