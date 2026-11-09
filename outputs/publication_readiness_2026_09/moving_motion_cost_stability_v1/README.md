# Does a Stable Cost Estimate Pick a Better Edit?

I refitted the pooled cost model while leaving out one fitting source at a
time. I then compared its lowest predicted gain with its mean prediction.
The two policies used the same forecasts, one edit in each of the same 1,613
queries per seed, and the same peak displacement in each query. The minimum
changed 393 recipients across the three seeds, or 8.1% of matched edits.

| Contrast | Observed gain (CV-normalized pp, 95% source bootstrap) | Full-grid bounds (pp) |
|---|---:|---:|
| Minimum over mean of source-deletion heads | +0.0149 [-0.0083, +0.0544] | [-0.1310, +0.1607] |
| Minimum over original full fit | +0.0184 [-0.0094, +0.0653] | [-0.1837, +0.2205] |
| Minimum over unedited OLS4 | +0.1150 [+0.0604, +0.1818] | [-0.9023, +1.1322] |

The primary lower-bound contrast has a source-bootstrap interval of
[-0.1945, -0.0810] pp. Minimum scoring lowered the observed positive-harm
mass against OLS4 by 1.33% versus the mean, but increased its missing-step
disagreement mass by 0.16%. These are descriptive sums over repeated
views, not independent-sample effects. None of the three arms passed the
original CV-referenced 2% rule in any of the 102 recording/seed views.
Deleting a source tests sensitivity; the minimum is not a calibrated
uncertainty bound. Unlike [group DRO](https://arxiv.org/abs/1911.08731),
the loss here is not a worst-group training objective.

## Why this edit budget cannot repair the risk

I also gave a hypothetical policy more power than any arm: it can pick any
one row per query after seeing the outcomes, erase that row's entire known
positive harm, ignore all missing-label harm, and use the largest possible CV
reference mass. For each subset, the resulting lower bound is
`(total known positive OLS4 harm - sum of the largest harm in each query) / total known CV error mass`.
It exceeds 2% on **33/34** recordings for all targets and **33/34** for easy
targets; the union is **34/34**. The respective worst lower bounds are 24.34%
and 153.38%. Each subset can remove a different row, making the calculation
even more optimistic. This proves failure under the specified OLS4 starting
forecast and one-edit budget on these recordings, not failure of all possible
CV-anchored policies or all larger budgets. The practical change is to keep
OLS4 as a strong accuracy comparator and test a genuinely CV-anchored joint
policy rather than relabel sparse OLS4 editing as safe deployment.

## Reproduction and scope

```sh
.venv-pytorch/bin/python scripts/run_m3w_motion_cost_stability.py
.venv-pytorch/bin/python scripts/run_m3w_motion_cost_stability.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_sparse_anchor_risk.py
```

`fresh_run`: 27 native Torch fits and 27,000 optimizer updates, taking
24.66 s locally (2.62 GB maximum resident memory), and a fresh optimistic
risk calculation. Training resumes from 100-step checkpoints.
`cached_verified`: frozen full-fit directions, original data roles and
cuts, and exact zero-update replay (10.30 s). 16 focused tests pass.
Source bootstrap: 3,000 draws after averaging three seeds, 34 exposed
recordings from 14 source groups. `not_run`: independent confirmation,
deployment, Stage5C and SMC. The 30 independent DroneCrowd groups stay sealed;
the original 2% rule is unchanged. Eight observed and 12 forecast native
annotation steps, local-image pixels, not seconds or metric coordinates.

[Protocol](protocol.md) | [Readout](readout.json) |
[Optimistic risk bound](sparse_anchor_risk.json) | [Freeze](freeze.json)
