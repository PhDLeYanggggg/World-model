# A Causal Recording Ledger Improves Utility, Not Full Risk Control

I kept the eight alternatives available until each query's budget decision.
The ledger starts at zero and updates only after a current decision, using
estimated all/easy excess from the frozen source-excluded cost model. For
each current query it chooses the best feasible zero, one or two distinct
switches. It can spend previously saved estimated budget without reading
later observations or realized labels. Exact search at the current query
does not imply globally optimal scheduling. Synthetic exhaustive tests and
suffix invariance check the decision rule; all real choices replay exactly.

| Policy on exposed A | Switches | Original passes / 102 | Easy over 2% | Undefined easy | Completion lower vs CV (pp) |
|---|---:|---:|---:|---:|---:|
| Filter then rank | 6,184 | 2 | 64 | 36 | +0.07746 |
| Full bank, query budget only | 6,184 | 2 | 64 | 36 | +0.08275 |
| Full bank, causal ledger | 6,184 | 3 | 65 | 34 | +0.10154 |
| Free-count ledger | 6,876 | 7 | 67 | 28 | +0.10516 |
| Frozen calibrated filter | 474 | 19 | 5 | 78 | +0.00004 |
| Calibrated free-count ledger | 679 | 17 | 10 | 75 | +0.00011 |

The first three have the exact same count in every query, not only in
total. The primary direct paired lower, causal ledger versus query budget,
is **+0.01725 pp [+0.00179, +0.03326]**. Versus the raw filter it is
**+0.02185 pp [+0.00833, +0.03619]**. The ledger spends prior estimated
slack in 1,631 of 4,839 seed-query views. No matched solver falls back to the
known feasible predicted witness. Observed relative-pair displacement
reduction versus query budget is -0.00326 pixels [-0.01405, +0.00662]; there
is no evidence of improved interaction consistency in this contrast.

The positive effect is specific to allocation utility on A. Known easy
risk exceeds 2% in 65 matched-ledger views (63 for the raw filter). It
cannot be explained solely by future-label missingness. Calibrated pooling
recovers 205 switches, but its small direct lower gain over the calibrated
filter (+0.00007 pp) accompanies fewer observed risk passes and no clear
gain over CV. Risk was not moved to a new denominator. An empty policy
remains unsupported by the unchanged test.

## Do Learned Candidates Add Anything Beyond Causal Alternatives?

I used the physical-only pointwise counts for both reoptimized ledgers,
allowing all five physical alternatives in one arm and the full eight
in the other. Both make exactly **2,580** switches. The full ledger's direct
paired completion-lower gain is **+0.01376 pp [+0.00345, +0.02687]**.
It selects learned trajectories on 1,264 switches (EqMotion 252, PCT 787,
kinematic self 225). The physical ledger passes 12/102 original views;
the full one passes 6/102 and has 50 easy violations versus 45. The benefit
is therefore not just an artifact of a weak, non-reoptimized physical
control, but it does not establish a safe learned upgrade. This comparison
uses different common counts from the primary ledger comparison.

Intervals use 3,000 paired resamples of 14 source groups after averaging
three actual forecast seeds. All 125,540 rows, 34 recordings and 102
repeated views are retained. Per-record ADE, FDE availability, tail error,
hard/easy metrics and undefined risks are in the scalar tables.

**Decision.** Preserve the chronology-correct ledger as a research
candidate. The open problem is whether estimated credit predicts realized
selected harm on new sources, not whether the current-query search is
large enough. No further tuning on A to claim deployment. Independent
calibration and untouched confirmation remain unrun. No metric/seconds,
physical safety, Stage5C or SMC claim.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_causal_record_ledger.py freeze
.venv-pytorch/bin/python scripts/run_m3w_causal_record_ledger.py decision_replay
.venv-pytorch/bin/python scripts/run_m3w_causal_record_ledger.py score
.venv-pytorch/bin/python scripts/run_m3w_causal_record_ledger.py replay
.venv-pytorch/bin/python scripts/run_m3w_ledger_physical_control.py freeze
.venv-pytorch/bin/python scripts/run_m3w_ledger_physical_control.py score
.venv-pytorch/bin/python scripts/run_m3w_ledger_physical_control.py replay
```

Registrations `1e094336` and `648bcd7b`. No model refits. Both matched
controls replay to their original actions and scalar scores. Main freeze,
action replay, score and score replay finish locally in 9.37/9.20/14.16/
14.33 s with peak RSS 1.64 GB, native arm64 and no worker processes.

[Protocol](protocol.md) | [Ledger metrics](analysis.json) | [Physical control](physical_control/analysis.json)
