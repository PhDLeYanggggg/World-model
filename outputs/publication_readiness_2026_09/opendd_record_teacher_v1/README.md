# Learning from Feasible Recording Assignments

I tested whether recording-level constraints should enter the training
labels rather than only evaluation. The held OpenDD clock, menu and
2,280 edits are identical for all arms. Only the chosen recipients
change. No thresholds were selected on held outcomes.

On donor recordings I use two fixed causal clocks, immediate release
and one-query waiting. Candidate forecasts and release heads exclude
the donor and outer site. With those counts fixed, a solver nominates
a slate assignment with positive lower gain over both CV and OLS4,
positive easy reference and the original 2% constraints. Each witness
is checked in the original scorer. The local teacher uses the same
donor queries and counts but nominates the largest per-query lower CV
gain. Unsupported donor views supply neither teacher with a label.

There were 5,921 donor clock views across repeated folds and seeds:
847 qualified, 1,435 received no credits, 1,013 lacked reachable easy
support, 2,510 were risk-infeasible, and 116 lacked positive dual gain.
From qualified views there were 4,742 choice examples, with teacher
labels differing on 446 (9.4%). These are training exposures, not
independent sample counts.

## Matched Held Comparison

| Policy | Edits | Undefined | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: |
| Fixed rank | 2,280 | 409 | 29 | 16 |
| Previous scalar regression | 2,280 | 399 | 30 | 20 |
| Previous pairwise harm guard | 2,280 | 410 | 30 | 16 |
| Local lower-gain teacher | 2,280 | 386 | 19 | 10 |
| Recording risk teacher | 2,280 | 383 | 23 | 15 |

The learned teachers differ on 569 recipients. Recording supervision
raises risk passes by four versus the local teacher but reduces
paired lower utility by **0.066 percentage points**, 95% CI
**[-0.144, -0.002]**. It also loses to the previous scalar head
(-0.199 pp [-0.318, -0.072]) and the harm guard
(-0.325 pp [-0.621, -0.110]). Against CV it gives -0.211 pp
[-0.621, +0.134]. It does not qualify for promotion.
Intervals resample seven sites after averaging three seed views,
using 3,000 draws. The 606 held views come from 202 exposed recordings.

## Can One Additional Edit Restore Support?

In a separate outcome-aware diagnostic I retain every existing edit
of the frozen clock policy and enumerate one extra recipient at any
query. It can use the original neural path or shrink its displacement
from CV to 0.5 or 0.25. Both original lower gains and the 2% risk
and support rules must pass; every positive witness is checked.

| Missing-support cohort | Views | Qualify at full amplitude | At 0.5 | At 0.25 | At any of the three |
| --- | ---: | ---: | ---: | ---: | ---: |
| No edits at the frozen clock | 244 | 125 | 135 | 138 | 139 |
| Edits, but no reachable easy reference in the fixed-count menu | 68 | 20 | 25 | 26 | 26 |

The amplitude columns overlap. Among these 312 views, 165 can
be rescued by one outcome-selected extra edit under the unchanged
risk rule. Another 30 have no legal easy candidate, 60 cannot pass
risk with one addition, and 57 lack dual gain even where risk passes.
This is an exact enumeration within a finite one-addition menu,
not a result at the original matched budget or a causal policy.
An online policy would not know at the start which recordings
will end without edits or easy reference.

The next useful comparison is a prospective single reserved slot
with past-only eligibility and a matched simple release control.
The slot would be tested as a separate development arm, with the
original 2% rule and fixed-budget experiment left intact. I would
not spend another round on imitating the same feasible assignments.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_record_teacher.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_record_teacher.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_one_extra_edit.py
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py --include-record-teacher
```

`fresh_run`: 42 local arm64 Torch fits and 21,000 optimizer updates,
36.7 seconds, plus the one-edit enumeration. Four threads, zero
loader workers, checkpoint and heartbeat every 100 steps; the same
command resumes. `cached_verified`: trajectory bank, release heads
and held clock; the 20.1-second zero-update replay matches the
frozen readout. Twenty-nine focused tests passed.

Native 8/12 annotation steps and dataset-local coordinates.
No new CREATE job, deployment, Stage5C or SMC. DroneCrowd's
15/30 roles remain frozen and confirmation outcomes were not opened.

[Registration](registration.json) | [Site and seed results](analysis.json) |
[One-edit registration](support_registration.json) | [Support capacity](support_capacity.json)
