# DroneCrowd: The Two-Switch Budget Cannot Reach OLS4

**2026-11-08.** Fresh outcome-aware diagnostic on the already opened
40 calibration recordings, 15 source groups, three frozen seeds. No
training, threshold choice, confirmation or deployment was performed.
The original calibration remains failed.

## What the bound answers

On each query, keep the actual joint arm's number of changed paths,
counting changes rather than nominations. Give an impossible oracle
the ability to reduce available-label ADE to zero on any chosen row.
An unchanged row contributes zero improvement; a changed observed row
cannot gain more than its original CV error. Wholly unknown rows cannot
add positive value to the registered conservative score. Therefore
the sum of the largest $K_q$ observed CV errors, divided by the number
of requested agents in query $q$, upper-bounds the score of **any**
forecaster with that intervention budget. It ignores all risk
constraints. It is a bound on the registered available-label score
with unknown-row debit, not on actual full-grid accuracy.

For a direct contrast to OLS4, subtract OLS4's observed gain and
omit the nonnegative unknown-disagreement debit. This upper-bounds
the original conservative paired contrast, not just a difference
between two lower bounds. Group aggregation and observed CV mass
normalization match the parent readout.

## Results

| Quantity | pp on observed CV error | 95% group-bootstrap interval |
|---|---:|---:|
| Frozen joint actual conservative gain vs CV (parent) | +1.766 | [+1.477, +2.106] |
| Zero-harm frozen menu, original query cap, gain vs CV | +3.171 | [+2.324, +4.230] |
| Perfect-switch upper bound, original query cap, gain vs CV | **+5.293** | [+3.846, +6.954] |
| All-agent OLS4 conservative gain vs CV | **+51.288** | [+46.487, +55.878] |
| Perfect-switch upper bound on the conservative contrast vs OLS4 | **-46.124** | [-51.760, -40.343] |
| Unrestricted zero-harm frozen menu, gain vs CV | +55.190 | [+51.528, +58.734] |

The perfect-switch CV bound is below the OLS4 lower score in all
15 source groups. It reproduces the original 67,440 changed paths across
the three seeds. This establishes a coverage limitation even if one
improves the selector and neural predictions. OLS4 changes many more
paths; this is **not** a claim that it wins an equal-budget contest.
The hypothetical perfect ceiling at uniform caps of 8, 16, 32 and 64
is +16.782, +28.996, +47.300 and +69.481 pp. Those are diagnostic caps,
not revised evaluation rules or proposed deployment settings.

For the finite-menu witness, choose the largest strictly positive
observed gain per row among OLS4 and the already frozen sparse policy
outputs. Wholly unknown rows retain CV. Without the query cap, the
witness changes 3,573,544 paths and passes the **original** rule in
120/120 views with all easy references defined. It uses future labels
to select, so it is not realizable at inference. Its gain uses the
registered query weights; its risk is checked by the original scorer. The capped menu
gain only describes zero-harm choices before the easy support
requirement; it does not assert a capped risk pass.

**Decision:** Retain OLS4 as a required accuracy comparator and the
registered CV risk reference as is. A new causal method needs broader
motion coverage if it is to compete with OLS4 on all-agent ADE, with
neural replacement tested as an increment rather than the only source
of coverage. This is a design direction, not evidence for a new policy.
The original 2% gate remains unchanged and confirmation stays sealed.

## Reproduction

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_perfect_switch_ceiling.py register
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_perfect_switch_ceiling.py score
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_perfect_switch_ceiling.py replay
~~~

The score phase reopens only approved calibration labels, verifies the
forecast freeze and checkpoints each completed recording. Replay reads
those checkpoints, not labels, and reproduces the output exactly.
Intervals use 3,000 group draws after seed averaging, not individual
windows. The OLS4 conservative readout and joint switch count match
the original result. [Registration](registration.json) |
[Machine-readable evidence](analysis.json).
