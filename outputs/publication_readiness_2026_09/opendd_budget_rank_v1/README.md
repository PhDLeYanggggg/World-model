# Ranking Slates Without Changing the Budget

I kept the original causal allocation for both edit roles. All
policies make 943 edits, with the same number of gain and support
edits at each query and the same peak displacement for each slot.
At a released query the policy can change which current track
receives an edit and choose its neural, OLS4 or observed-prefix
direction. Alternative directions are shrunk to the anchor radius,
never amplified. The bounded menu includes the anchor and up to
three other nominated rows per role, with distinct recipients.

The models share the same fitting rows and optimizer updates.
The regression arm predicts the observed paired gain, missing-step
debit, risk changes, and joint feasibility. The rank arm adds a
preference against the original slate only when the observed gain
interval has a definite sign. The geometry control retrains the
rank model after masking four relative-path features of the current
scene. The guard uses estimated risk changes and may retain the
original slate. These are not realized-risk certificates.

The donor nomination and forecast heads exclude the fitting donor
and the evaluation site. Future observations enter labels and
scoring only. At an individual training query, the recording label
replaces that query's slate and holds other edits fixed. The
online policy has no future slate input and the full recording is
rescored after all causal choices.

| Policy | Joint passes | Risk passes | Undefined |
| --- | ---: | ---: | ---: |
| Frozen allocation and original edits | 138 | 172 | 295 |
| Point regression | 101 | 138 | 319 |
| Ranking | 115 | 153 | 313 |
| Ranking without relative-path features | 97 | 134 | 327 |
| Guarded ranking | 126 | 165 | 298 |

The primary conservative contrast, rank versus point, is
**-0.00010 pp [-0.00594, +0.00598]**. Rank versus the frozen
policy is +0.00053 pp [-0.03495, +0.03620]. Geometry versus its
retrained ablation is +0.02707 pp [+0.00146, +0.05375], but does
not establish an advantage over the original policy or physical
safety. Relative-trajectory error gain for rank versus point is
-0.00292 pp [-0.00825, +0.00093]. The held menu contains 269
queries with a locally feasible improving swap out of 891; this
is not an oracle result that can be composed across the recording.

## Why Did Reference Support Fall?

The post-result replay keeps only one kind of swap at a time,
with no refitting. Retaining just the 123 shape-only changes gives
132 joint passes and the same 295 undefined views as the original.
Retaining the 289 queries with at least one recipient change gives
121 joint passes and 313 undefined views. The latter loses 47
previously defined easy references and gains 29. All such changes
in the full ranker are explained by changed recipients. Shape-only
choice can still cause harm or lose passes without changing
whether risk is estimable.

The results use three seeds over seven sites; intervals resample
sites 3,000 times after averaging seeds. The 606 recording-seed
views are exposed development evidence. No policy is promoted.
The next test should preserve the original support recipient and
learn on the gain role separately rather than add a new term to
the same unrestricted ranker.

The [decision-focused ranking literature](https://proceedings.mlr.press/v162/mandi22a/mandi22a.pdf)
already distinguishes value regression and ranking solution
pairs. Our comparison applies that distinction to incomplete
trajectory labels and matched intervention budgets; it is not
a general ranking innovation or a proof of deployment safety.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_budget_rank.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_budget_rank.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_budget_rank_attribution.py
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_calibration_roles.py verify
```

`fresh_run`: 63 local arm64 Torch fits and 31,500 optimizer
updates, four threads and no data-loader workers. Checkpoints and
heartbeat are saved every 100 steps. `cached_verified`: frozen
allocation, donor forecasts and nomination heads, plus zero-update
checkpoint replay. `not_run`: independent DroneCrowd confirmation.
The 15/30 split and 2% recording rule are unchanged. No new CREATE
job, Stage5C, SMC, deployment or paper submission.
