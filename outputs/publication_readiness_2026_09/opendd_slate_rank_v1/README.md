# Learning Which Recipient Set Helps

I froze the previous OpenDD policy's clock, its 2,280 realized edits,
and the available recipient sets. Every arm uses the same number
of edits at each query. Only recipient identity can change. Donor
trajectory and release heads exclude both the donor and the outer
evaluation site. I fit all same-count pairs reachable under the
original release window; I do not fit a clock on held outcomes.

For two sets, the observed gain difference is charged the missing
step debit only on recipients that differ. The training target is
the signed advantage outside that uncertainty interval, zero if
the comparison is ambiguous. Three separate outputs predict
differences in all excess, easy excess and signed easy risk.
The guarded arm keeps the fixed-rank set unless another set has
positive predicted gain and no predicted increase in any of the
three quantities. This is a selection rule, not a risk certificate.
The scalar control learns the previous additive proxy on the same
features and pair exposures.

## Main Comparison

| Policy | Edits | Changed recipients vs fixed | Undefined | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed rank | 2,280 | 0 | 409 | 29 | 16 |
| Previous joint model | 2,280 | 1,048 | 401 | 26 | 16 |
| Scalar regression | 2,280 | 1,077 | 399 | 30 | 20 |
| Pairwise gain + harm heads | 2,280 | 1,116 | 402 | 23 | 15 |
| Pairwise with predicted harm guard | 2,280 | 487 | 410 | 30 | 16 |
| Per-query lower-gain oracle | 2,280 | 1,122 | 420 | 64 | 51 |

The oracle is an outcome-aware accuracy control against fixed
rank, not the earlier exact joint-risk optimum (107 views at the
same clock). It does not impose the recording risk rule.
Joint pass requires both lower gains over CV and OLS4 and the
original 2% risk and support checks.

| Paired lower gain | pp [95% site-bootstrap interval] |
| --- | ---: |
| Pairwise vs fixed rank (registered primary) | +0.107 [+0.014, +0.225] |
| Guarded vs fixed rank | +0.218 [+0.031, +0.540] |
| Scalar vs fixed rank | +0.051 [-0.052, +0.161] |
| Pairwise vs scalar | -0.006 [-0.064, +0.051] |
| Guarded vs scalar | -0.012 [-0.186, +0.147] |
| Guarded vs CV | +0.004 [-0.211, +0.212] |
| Oracle vs fixed rank | +1.067 [+0.481, +1.558] |

The seven-site bootstrap averages three seed views before each of
3,000 draws. The 606 views come from 202 exposed recordings. The
ranker gets 3,028 of 5,488 outcome-resolved pairs right (55.2%),
versus 2,968 for scalar regression (54.1%); pairs are diagnostic,
not independent observations. The guard reduces the number of
recipient changes; the macro mean easy degradation is 1.57% for
the guard versus 2.24% for fixed rank. This is not per-recording
compliance: 410 views remain undefined and only 30 pass risk.

## Site Sensitivity And Decision

The guard improves on fixed rank at five sites and ties at two.
The biggest site contributes +1.155 pp. Omitting it leaves
+0.061 pp [+0.025, +0.097] for the guard and +0.060 pp
[-0.003, +0.138] for pairwise without the guard. Weighting sites
by observed CV error mass rather than equally gives +0.090 pp
for the guard over fixed rank and -0.028 pp versus CV. These are
post-hoc robustness checks, not new selection criteria.

Direct ranking improves the fixed-rank comparison but has not
shown a benefit over the trained scalar control, the CV baseline
or the full-roster risk requirement. I keep the existing deployment
decision. The next experiment should align supervision with the
recording-level constraints, rather than assuming that an improved
within-query ranking meets them. The frozen clock itself lacks
edits or reachable easy support on many recordings, a separate
limitation no ranker can remove at fixed counts.

Pairwise training itself is not the contribution. [Mandi et al.](https://proceedings.mlr.press/v162/mandi22a.html)
already formulate decision-focused learning through ranking feasible
solutions, and [Mao et al.](https://proceedings.mlr.press/v235/mao24d.html)
study learned deferral for regression. My empirical question is
whether an interval-resolved recipient comparison at an identical
edit clock transfers to the original recording-level risk checks.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_slate_rank.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_slate_rank.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_slate_rank_sensitivity.py
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-rank
```

`fresh_run`: 42 local arm64 Torch fits and 21,000 optimizer updates,
31.4 seconds, plus site sensitivity. Four Torch threads, zero
loader workers; checkpoint and heartbeat every 100 steps. The
same command resumes. `cached_verified`: existing forecast bank,
release heads, and realized clock; an 18.3-second zero-update replay
matches the immutable result. Twenty-six focused tests passed.
No CREATE job, deployment, Stage5C, or SMC. No DroneCrowd
confirmation outcome was accessed. Native 8/12 annotation steps
and dataset-local coordinates; no metric or seconds claim.

[Registration](registration.json) | [Site and seed results](analysis.json) |
[Sensitivity](sensitivity.json)
