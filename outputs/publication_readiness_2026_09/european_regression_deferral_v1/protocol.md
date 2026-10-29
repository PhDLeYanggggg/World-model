# Published Regression-Deferral Surrogate Control

## Material Passport

Prospective SOURCE-development baseline,29October2026. Implement the two-stage
logistic cost-sensitive surrogate of Mao,Mohri,Zhong,ICML2024,Section4,Eq.(3),
on the same frozen forecasts and source partitions. This is an application
adapter,not a reproduction of their tabular experiments or proof of their
assumptions for our native partial-label ADE.
[Proceedings](https://proceedings.mlr.press/v235/mao24d.html),
[original paper](https://cs.nyu.edu/~mohri/pub/regdef.pdf).

## Objective

Treat the unchanged causal floor as expert0 and the six motion/original-neural
forecasts as remaining fixed experts. For each available bank,known training
costs c_j are native ADE divided by the frozen fitting-source cost scale. With
base2 logistic loss,Eq.(3) reduces to

```
sum_j (sum_l c_l - c_j) * (-log2 softmax(score)_j).
```

There is no one-hot oracle winner,per-row cost normalization,consultation fee,
ground-truth inference feature or new trajectory training. Unknown future
outcomes are not regression labels and are not sampled by the optimizer;
they remain in inference and completion-bound evaluation. This native partial
ADE application does not establish the original bounded-loss consistency
assumptions or any2%risk guarantee.

## Matched Scope and Training

Keep the six producer/controller contexts,forecaster seed43,24verified SOURCE
TRAIN packets,whole-record fitting/inner-held partitions,easy event,masks,
floor and rollout envelopes. Train physical-only,physical+neural and neural-only
controllers with the same three head seeds,2,000updates,source-balanced128row
batches,physical-fitting feature normalizer and25k-scale conditional MLP.
Fitting row draws must match the preceding moment learners exactly.

Each class has380causal geometry/rollout features and an8-way expert code.
This includes a floor forecast feature row computed from past-only geometry.
Only the active bank enters each training loss;physical-only training must
not consume neural labels/features. Natural physical aliases of the floor
are merged into floor at inference,not counted as fake real interventions.

Report both the unconstrained minimum-surrogate-cost choice and a fixed
cost-gated version using the preceding frozen predicted2%all/easy conditions.
The latter is a fixed control,not a post-readout cutoff search. Full-policy
safety remains evaluated at2%on every required recording;empty risk is
not_estimable. Count-matched controls are utility-only.

## Run and Claim Boundaries

Use native arm64Torch CPU4/interop1,num_workers0,checkpoints100,heartbeat50.
Run all-arm20update exact-resume pilot first,then all54fits/108,000updates;
freeze endpoints before readout. Preserve10GiB and up to12h healthy runtime.
Reuse the paired locality/bootstrap readout and retain all seeds/recordings.

The comparison is a published-loss adapter with fixed experts,not a claim that
our new routing method is novel. No independent role,metric/seconds,true3D,
foundation,Stage5C orSMC execution/claim is enabled.
