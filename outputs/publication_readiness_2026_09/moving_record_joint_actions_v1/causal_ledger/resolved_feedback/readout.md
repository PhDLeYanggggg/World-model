# Can Past Forecast Errors Tell Us When To Switch?

I tested whether a forecast that has already come due helps identify a
good switch now. At frame t I compare the forecast issued at t-12 with
the same agent's position observed at t. The comparison is only between
observed frames; there is no access to the new forecast's future. The new
cost heads use either that agent's latest resolved forecast or that
feedback plus a decayed summary of the same recording. They do not
modify any candidate trajectory.

On the exposed Moving A development roster (125,540 rows, 1,613 queries,
34 recordings, 14 sources), there are 74,821 exact same-agent matches
across 1,256 queries. Twenty-six recordings have matches. Carrying
resolved feedback forward gives 75,925 rows with own feedback and
101,269 with recording feedback. I retain all rows, including those
without a match. The forecast banks for seeds 17/29/43 are verified
cache, fitted outside A. The 84 actionwise cost-head fits and all
feedback, allocation and scoring in this experiment are fresh. Each
head excludes its evaluated source from fitting.

| Compared With Corresponding Static Control | Known Utility Difference (pp) | Completion Lower Difference (pp) | 95% Source Interval For Lower | Switches | Original Risk Passes / 102 |
|---|---:|---:|---:|---:|---:|
| Own, matched | -0.00323 | -0.00785 | [-0.01330, -0.00287] | 6,184 vs 6,184 | 1 vs 3 |
| Own + recording, matched (primary) | +0.00361 | +0.00003 | [-0.01215, +0.01426] | 6,184 vs 6,184 | 1 vs 3 |
| Own, free | +0.04475 | +0.03228 | [+0.00170, +0.07517] | 7,187 vs 6,876 | 6 vs 7 |
| Own + recording, free | +0.02360 | +0.01013 | [-0.04685, +0.06582] | 6,710 vs 6,876 | 6 vs 7 |

The matched arms have identical switch counts at each query and identical
predicted budget costs and legal actions. They change the ranking signal,
not the risk allowance. The two static controls reproduce the prior
ledger decisions exactly. The free arms can change both decisions and
counts. Intervals use 3,000 paired bootstrap draws over sources after
averaging the three banks, not independent agent-row resampling. The
completion lower comparison conservatively accounts for wholly unknown
outcomes under the existing protocol.

**The primary comparison does not pass.** Adding recording feedback changes
5,034 row decisions across the three banks, but does not improve the
matched utility reliably. On the same original selected rows the
radius-normalized easy excess MSE falls from 0.02238 to 0.02155, and
mean bias changes from -0.05048 to -0.03251: better average prediction,
but still underestimation. Under the unchanged observed gate, 65 of 102
matched recording views remain above the 2% easy completion upper limit
and 34 have undefined easy risk. The free own-feedback arm shows a small
utility signal, not a controlled ranking gain or a safety improvement.
This isolates a decision error, not a lack of trajectory diversity or
proof that all possible feedback models fail.

No independent outcomes were accessed. This is image-coordinate 8-to-12
annotation-step evidence, not metric or seconds-level forecasting.
Deployment, original risk and support rules, Stage5C and SMC are unchanged.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_resolved_motion_feedback.py features
.venv-pytorch/bin/python scripts/run_m3w_resolved_motion_feedback.py fit
.venv-pytorch/bin/python scripts/run_m3w_resolved_motion_feedback.py score
.venv-pytorch/bin/python scripts/run_m3w_resolved_motion_feedback.py replay
```

Registered at `e14e6e81`. Local native arm64 execution uses no worker
processes, with atomic per-seed checkpoints. Features (3.59 s) freeze
before opening fitting targets; fit and allocation (18.68 s) freeze
before scoring. Scoring and replay (9.42 / 9.29 s) match the full
analysis SHA-256 exactly. Twenty-seven focused tests pass. Full per-view
metrics and array caches stay local; the repository keeps this aggregate
readout, [protocol](protocol.md) and [score replay receipt](replay_execution.json).
