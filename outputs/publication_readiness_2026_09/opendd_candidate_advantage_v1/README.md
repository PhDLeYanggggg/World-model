# Direct Candidate Advantage With Double Exclusion

Predicting CV difficulty did not reliably identify when a frozen neural
forecast improves it. I trained a head on each candidate's own gain
and harm instead. For every outer and fitting-site pair, the trajectory
producer excludes both sites, so the head learns from predictions made
out of source. Evaluation still uses the frozen six-source ego-only
EqMotion forecaster. See the
[protocol and commands](protocol_and_reproduction.md).

## Equal-Count Results

The previously registered difficulty rule sets every query's count.
Each active policy uses the same candidate path for a given row, and
each makes 3,323 edits. The row assignments and total peak displacement
can differ. Gain and gain-plus-risk use the same head; the latter
penalizes positive predicted all, easy, and signed-easy excesses.

| Policy | Easy rows selected | Undefined views | Original risk passes | Joint passes / 606 |
| --- | ---: | ---: | ---: | ---: |
| CV, no edits | 0 | 606 | 0 | 0 |
| CV difficulty | 454 | 437 | 12 | 6 |
| Motion disagreement | 472 | 438 | 11 | 5 |
| Learned gain | 546 | 401 | 11 | 7 |
| Learned gain and risk | 564 | 401 | 13 | 9 |
| Outcome-aware utility only | 415 | 452 | 38 | 35 |

Joint passes require the original 2% risk checks **and** positive
conservative utility against CV and OLS4. There are 202 recordings,
repeated over three training seeds, not 606 independent observations.
Effects are direct paired lower gains in percentage points of the
common CV error mass. The 3,000 descriptive bootstrap draws resample
seven exposed development sites after seed averaging.

| Direct effect | Gain, pp [95% interval] |
| --- | ---: |
| Learned gain vs difficulty | +0.040 [-0.381, +0.434] |
| Learned gain and risk vs difficulty | +0.093 [-0.364, +0.504] |
| Learned gain vs CV | -0.154 [-0.493, +0.200] |
| Learned gain and risk vs CV | -0.086 [-0.411, +0.254] |
| Outcome-aware utility rank vs CV | +1.336 [+0.507, +2.130] |

The actual positive-gain AUPRC is 0.600 for learned gain and 0.596 for
learned gain and risk, versus 0.555 for CV difficulty (prevalence 0.480).
This ranking diagnostic does not establish policy superiority. The
risk readout reduces the sum of peak deviations from 2,370.09 to
1,599.29 local units versus difficulty, despite equal counts, and
increases reference support. Neither readout reliably lowers error.
Paired relative-path effects versus difficulty are -0.309 pp
[-1.086, +0.389] for gain and -0.264 pp [-1.041, +0.468] for risk.
These are errors in relative position, not physical-safety measurements.

## What The Same Counts Can Achieve

The integer-program diagnostic optimizes the lesser of the two
conservative gains under the original recording risk rule, using the
same candidate paths and exact count for every query. Every positive
witness reproduces in the original scorer; no cases are unresolved.

| Exclusive outcome | Views |
| --- | ---: |
| No edits | 217 |
| No eligible easy reference at the active queries | 85 |
| Original risk infeasible at the exact counts | 208 |
| Risk feasible, but cannot strictly beat both references | 8 |
| Qualifies with outcomes known | 88 |
| Total | 606 |

The 88-view ceiling is for this path bank and schedule, not the dataset.
The accuracy-only oracle does not optimize the risk gate. The comparison
shows both a causal ranking gap and a budget/support limitation. I do
not replace the existing policy or use this schedule for calibration.
Before further ranking experiments, the next intervention policy should
show that its causal allocation can reach enough reference support
under the unchanged 2% rule.

## Evidence

- `fresh_run`: 63 double-excluded EqMotion producer fits and 21
  selector fits, 60,900 Torch updates, paired readout, and original-rule
  integer-program diagnostic. Total first run: 15.8 minutes locally.
- `cached_verified`: frozen outer forecasters and difficulty rule,
  regenerated outputs from all checkpoints, capacity witnesses and
  immutable results. Zero-update replay took 57.6 seconds.
  22 focused tests passed.
- `not_run`: new DroneCrowd calibration, independent confirmation,
  deployment, Stage5C and SMC. Existing 15/30 roles unchanged.

[Readout](analysis.json) | [Capacity](capacity.json) |
[Registration](registration.json) | [Producer receipts](producer_freeze.json) |
[Selector receipts](head_freeze.json)
