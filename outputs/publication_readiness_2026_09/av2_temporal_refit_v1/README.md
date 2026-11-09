# AV2: Matched Temporal Refit Recovers Some Choices, Not Transfer

The [temporal menu experiment](../av2_temporal_menu_v1/README.md)
expanded the number of qualifying alternatives to a fixed
independent choice from 48 to 61. The frozen scorer performed
worse on the bigger bank. I tested whether giving the scorer
examples of the new timing profiles fixes that gap.

## Design

Both action scorers start from the same four-donor checkpoint.
Each sees the same scenes, actor, edit slot and source forecast
family per update. The original-bank arm sees the original
action; the temporal arm samples original, early or late
timing uniformly for the matched action. Both use the same
five independent-action gain and risk targets and easy
membership loss as the original control. Both train for
320 steps, batch 96, on cached out-of-city forecast inputs.

I decode each trained scorer on each bank, so extra updates
and access to the wider option bank are controlled separately.
The same two edits and peak displacements, 8/12 grid, city
roles, seeds 17/29/43 and original 2% rules are used. The
calibration test is for a reference alone, not a full joint
deployment: any nonempty prefix must pass both causal floors
and all three original risk tests for every released view.
Calibration labels come from a city excluded from training.
Held choices, scores and thresholds are frozen before scoring.

## Held Development Results

Counts are over the same 1,020 seed views (340 scenarios).
The first two rows are frozen and replayed exactly.
“With alternative” means there is a qualifying joint path
in the same decoding bank given the independent choice;
it is not a selected joint prediction.

| Training | Decode | Eligible reference | With alternative | Eligible / released |
|---|---|---:|---:|---:|
| Old | Original | 63 | 48 | No release |
| Old | Union | 50 | 42 | No release |
| Original refit | Original | 68 | 48 | 0 / 2 |
| Original refit | Union | 49 | 40 | No release |
| Temporal refit | Original | 66 | 46 | 0 / 2 |
| Temporal refit (primary) | Union | 59 | 54 | 0 / 3 |

At the same union decoding bank, temporal training recovers
**+0.73 pp [0.16, 1.51]** of eligible references and **+1.20 pp
[0.46, 2.15]** of views with an alternative vs original-only
refitting. This diagnoses a training-distribution effect.
The primary does not beat the original refit on eligibility
**(-1.07 pp [-1.63, -0.27])**. The difference in views with
an alternative vs that comparator is **+0.15 pp [-1.00, 1.49]**.
Three thousand paired city draws average seeds within scenarios
first. The intervals are development uncertainty, not
independent confirmation.

The primary's forced trajectory lower gain against the
old original-bank scorer is **-3.61 pp [-3.97, -3.24]**.
In Palo Alto it still produces zero eligible references.
Three views release across Palo Alto and Dearborn; none
qualifies under the unchanged rules. Missing edited-easy
support decreases from 631 to 529 vs the old union scorer,
but more support does not ensure that all risk and gain
requirements hold together.

The wider menu helps with hindsight and part of that gap
is learnable, but this refit does not yield a better
deployable reference or a joint policy. The next comparison
should keep the original causal reference as a selectable
incumbent, then ask whether a source-separated replacement
rule adds value at the same budget. Do not retune these
held thresholds or make a generalization claim from this study.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_av2_temporal_refit.py fit
.venv-pytorch/bin/python scripts/run_m3w_av2_temporal_refit.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_av2*.py \
    tests/test_m3w_interaction_frontier.py tests/test_m3w_slate_relative_gain.py
```

36 new fits across 18 fold-seed combinations. Native arm64
PyTorch, four compute threads, one process, no CREATE job.
Training plus freeze took 42.09 s at 0.69 GB max RSS; scoring
took 7.93 s. Checkpoints every 40 updates retain optimizer and
sampler state; repeating `fit` resumes. PID and step are in
the local heartbeat. All 68 focused tests passed. No reserved
outcomes, manuscript, policy, Stage5C or SMC changes.

[Registration](registration.json) · [Freeze](freeze.json) · [Metrics](summary.json)
