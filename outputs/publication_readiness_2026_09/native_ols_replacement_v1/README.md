# Learning when to replace OLS4 from past-only diagnostics

DroneCrowd calibration showed that a CV-centered selector can beat CV yet lose to a stronger reference. I moved the training target to OLS4 on nine existing fitting groups, without reading DroneCrowd calibration or confirmation data. Each arm uses the same four causal forecast options and the same per-query intervention counts. Three paired seeds, three learned arms, and 600 updates per fit produced nine fresh Torch models. The full 125,540-window development roster was held out from fitting, previously exposed, and scored only after choices were frozen.

| Fixed-count comparison | Observed-label gain (pp) | Completion lower (pp) | 95% paired source CI for lower (pp) |
| --- | ---: | ---: | ---: |
| OLS hindcast vs unmodified OLS4 | +0.333 | -1.753 | [-3.729, -0.491] |
| OLS motion vs unmodified OLS4 | +0.123 | -1.960 | [-4.059, -0.455] |
| CV hindcast vs unmodified OLS4 | +0.127 | -1.927 | [-4.123, -0.521] |
| Unlearned hindcast vs unmodified OLS4 | +0.158 | -2.187 | [-3.767, -0.961] |
| OLS hindcast vs OLS motion | +0.210 | -1.311 | [-2.017, -0.693] |
| OLS hindcast vs CV hindcast | +0.205 | -3.049 | [-6.783, -0.825] |
| OLS hindcast vs CV | -3.626 | -16.134 | [-29.842, -6.204] |

The common denominator is the available CV ADE mass, not the OLS4 error mass. Each contrast charges its own forecast disagreement on wholly unobserved rows; lower contrasts are not additive. Seeds are averaged within source before 3,000 paired resamples of 14 source groups. All active arms make 9,675 OLS4-relative switches. OLS hindcast harms 3,683/8,871 observed switched rows, versus 4,606/9,302 for CV hindcast, but the former selects 804 wholly unobserved rows versus 373 for the latter. These are repeated windows, not unique people. The original CV risk diagnostic gives no passes for any OLS4-floor arm and is not relabeled as an OLS4 gate.

The short-history probes improve the known-label contrast but do not identify which full-horizon replacements are robust. On the [fixed-direction shrinkage diagnostic](../native_ols_replacement_amplitude_v1/README.md), every arm has its best common amplitude at zero. Selectively suppressing particular harmful or unobserved directions matters more than scaling every chosen direction down together. The rowwise hindsight headroom does not demonstrate that this choice can be learned from past features.

`fresh_run`: fits, frozen choice generation, score and exact replay. `cached_verified`: development roster and fitting lineage. No threshold search, new external outcome, metric/seconds claim, deployment, Stage5C or SMC.

```sh
runner=scripts/run_m3w_ols_replacement.py
.venv-pytorch/bin/python "$runner" train
.venv-pytorch/bin/python "$runner" infer
.venv-pytorch/bin/python "$runner" score
.venv-pytorch/bin/python "$runner" replay
```

Interrupted fitting resumes with `train --resume`. [Protocol](protocol.md) | [Results](analysis.json) | [Replay](replay_execution.json)
