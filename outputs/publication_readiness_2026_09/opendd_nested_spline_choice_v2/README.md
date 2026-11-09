# Can the spline help at the same intervention locations?

I kept the controller's 1,457 intervention slots fixed across 606
recording-by-seed views on seven OpenDD development locations. Each
comparison uses the same agent, starting time, and peak displacement
from constant velocity. The learned head chooses the original analytic
shape or the spline; it cannot change who is edited or when.

I used three seeds. For each outer location, the calibration candidate
and the model that scores it both exclude that location and the relevant
calibration location. Forecasts used as fitting inputs also exclude the
location of the row being predicted. This required 168 new forecast
fits and 84 new cost-head fits (168,000 total updates). The outer
six-donor forecasts and the controller slots were reused after hash and
role checks. No outer outcome was used to choose its rule.

| Path rule | Spline choices | Original 2% risk passes | Risk + both accuracy gains |
| --- | ---: | ---: | ---: |
| Original analytic shape | 0 | 112 | 77 |
| Past-prefix shape rule | Not applicable | 124 | 90 |
| Always spline | 1,457 | 92 | 71 |
| Learned gain only | 281 | 116 | 88 |
| Independent risk | 4 | 112 | 77 |
| Joint risk | 4 | 112 | 77 |
| Joint choice at the independent rule | 4 | 112 | 77 |

The primary contrast, joint risk versus past prefix, is **+0.008
conservative percentage points** (95% location bootstrap interval
**[-0.025, +0.059]**, 3,000 resamples after averaging seeds). Against the
original shape it is **-0.010 pp [-0.024, 0.000]**. Independent and joint
risk made identical choices, including when sharing the same rule. This
does not support a joint-selection advantage. The 334 undefined easy
references are unchanged and do not count as passes. Worst observed
signed easy degradation was **11.28%** for both the joint and prefix
rules; signed easy loss is only one part of the original risk check.

## Why the guarded rule barely intervenes

All seven calibration folds contained a positive-gain joint grid entry,
but only two folds chose a nontrivial rule. Of 76 positive-gain joint
entries, 56 violated the easy-excess constraint and only 19 satisfied
all constraints. Constraint violations can overlap. The calibration
roster chooses up to two moving recipients per query from history and
prefers the middle radius. This is not the actual streaming controller
roster: the two largest held locations together contain 1,395/1,457
actual slots across seeds, and neither selected a nontrivial risk rule.
These diagnostics identify easy risk and roster mismatch as priorities;
they do not establish that matching the roster will fix the policy.

Next I will generate calibration slots by replaying the same causal
controller with outer and calibration locations excluded, then test a
matched-count comparison on development data. The original recording
rule stays fixed. The result here does not change deployment or support
a physical safety, metric, seconds, 3D, or foundation-model claim.
DroneCrowd's 30 confirmation groups remain sealed; Stage5C and SMC are off.

## Reproduction

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/finalize_m3w_opendd_nested_spline_choice.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_nested_spline_choice.py
```

Without `--replay`, incomplete fits can resume. The first runner stopped
at JSON serialization after training, before outer scoring; the
[receipt repair](receipt_repair_v2.json) only converts NumPy booleans to
native JSON values. It restores the complete checkpoints and checks all
saved outer predictions exactly before scoring. The earlier inference
scope error is recorded in the [protocol](protocol.md).

[Registered design](registration.json) | [Freeze](freeze.json) |
[Readout](readout.json) | [Diagnostics](diagnosis.json) |
[Manuscript](../evidence_manuscript_v150/README.md)
