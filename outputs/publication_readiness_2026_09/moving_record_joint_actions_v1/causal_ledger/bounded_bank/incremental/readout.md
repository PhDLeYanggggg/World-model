# Direct Amplitude Costs Do Not Improve The Matched Ranking

I kept all 6,184 selected rows and their forecast families from the
expanded bank. The new head can only reduce each original amplitude.
I fitted the gain and harm of each adjacent amplitude increase directly
instead of subtracting separately predicted absolute costs. This uses
84 new source-excluded fits and three verified forecast seeds; it is
not a trajectory backbone refit. The direct and absolute-derived
policies use the same decision score. A second absolute-derived policy
also matches the direct policy's exact level multiset at every query.

| Policy | Full / Half / Quarter | Completion Lower vs CV (pp) | Easy Risk Above 2% | Undefined Easy | Original Passes / 102 |
|---|---:|---:|---:|---:|---:|
| Original expanded bank | 4,436 / 1,464 / 284 | +0.11837 | 66 | 36 | 0 |
| Absolute-derived attenuation | 4,329 / 1,505 / 350 | +0.11782 | 63 | 36 | 3 |
| Direct incremental attenuation | 3,842 / 1,742 / 600 | +0.11886 | 60 | 36 | 5 |
| Absolute-derived, exact amplitude match | 3,842 / 1,742 / 600 | +0.11908 | 61 | 36 | 4 |
| Quarter on the same rows and families | 0 / 0 / 6,184 | +0.06058 | 44 | 36 | 21 |

**The primary matched comparison does not pass.** At identical per-query
levels the direct model changes 368 row decisions across 384 exchangeable
query views. Its direct paired completion-lower contrast is -0.00022 pp
[-0.00531, +0.00378]. Against the original expanded bank the direct paired
lower is -0.00234 pp [-0.00597, +0.00029], even though the separate
lower-vs-CV scores rise slightly. The paired lower uses disagreement
between the two policies for missing outcomes and cannot be recovered by
subtracting separate lower bounds. Intervals use 3,000 paired source
draws after averaging the three banks.

The direct model is less negatively biased for added easy harm
(-0.01076 vs -0.01977 when normalized by full-path distance from CV).
But its mean squared error is higher for all three targets: gain
0.04173 vs 0.03350, all harm 0.00637 vs 0.00593, and easy harm
0.00454 vs 0.00386. Across all 102 views the all/easy completion risk
upper never increases under downward-only choices, as expected from the
fixed selected rows and convex error. This does not imply that the
original 2% checks pass or that signed easy degradation must improve.
The remaining 36 undefined easy risks stay undefined. The quarter
control here has different selected rows from the earlier quarter
control that passed 29 views; 21/102 is the same-row result in this study.

## How Much Can The Matched Assignment Recover?

As a posthoc diagnostic, I evaluated the only other feasible amplitude
assignment in each exchangeable query, without changing the selected
rows, families, or amplitude counts. An oracle using observed outcomes
finds 173 swaps with positive completion-lower gain. Even taking all
such swaps improves the full-roster lower against the direct policy
by only +0.00672 pp [0.00134, 0.01363]. This is the maximum in this
restricted two-way comparison under the existing lower objective;
it is not an upper bound on true unobserved gain, a safety oracle,
or evidence for a learnable policy. No label-aware actions are exported.

**Decision.** Small changes help reduce observed harm, but the gain in
passes mostly follows lower exposure, not a better amplitude ranker.
Prioritize identifying which agents should switch, particularly where
the model underestimates risk. Do not promote either amplitude policy.
This is exposed development evidence, eight observed to twelve future
annotation steps in image coordinates. No independent outcomes, rule
changes, deployment, Stage5C or SMC.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_incremental_amplitude.py fit
.venv-pytorch/bin/python scripts/run_m3w_incremental_amplitude.py score
.venv-pytorch/bin/python scripts/run_m3w_incremental_amplitude.py replay
.venv-pytorch/bin/python scripts/audit_m3w_incremental_amplitude_ceiling.py
```

Registered at `8877d6a4` before fitting. Native arm64 local fit takes
11.45 s with per-seed resume; score and exact scalar replay take
8.30 / 8.15 s approximately. Forty-three scoped tests pass.
[Protocol](protocol.md) | [Score replay](replay_execution.json) | [Diagnostic ceiling](ceiling.json)
