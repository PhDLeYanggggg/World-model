# Nested Calibration Preserves Accuracy, Not Recording-Level Safety

The self-history forecast retains a small accuracy advantage after
location-held calibration and beats two old-bank controls on the exact
same recipients. **Neither calibrated policy meets the unchanged
recording-level risk rule.** This is OpenDD development evidence;
DroneCrowd confirmation remains sealed.

## Nested Comparison

I trained 63 fresh five-location inner forecasters and reused 21 verified
outer checkpoints. Each alpha is chosen without any data from the outer
location. The uniform policy learns one coefficient; the agreement cap
shrinks that coefficient on targets where the neural prediction is farther
from CV than OLS4 is. All policies below edit the same 30,075 target rows
across three seeds, using a common eligible roster. The fixed 25% result
therefore differs slightly from the previous full-roster result.

| Policy | Conservative gain vs CV, pp [95% site bootstrap] | Worst signed easy ADE | Worst positive easy harm | Flight-seed risk passes |
|---|---:|---:|---:|---:|
| Fixed 25% | +1.810 [+1.068, +2.874] | +5.24% | 10.38% | 86 / 606 |
| Nested uniform | **+0.474 [+0.304, +0.735]** | **+0.43%** | 2.46% | **450 / 606** |
| Nested agreement cap | +0.549 [+0.349, +0.838] | +0.70% | 2.51% | 418 / 606 |

Both calibrated arms improve conservative gain at all seven held locations
after averaging seeds. Uniform alpha is 0.05 in 17 views and 0.075 in four.
Observed ADE/FDE are 0.6051/1.3467 for uniform, versus 0.6083/1.3534 for CV,
in published OpenDD units. Conservative gains include missing-step forecast
disagreement and are normalized by observed CV error mass. The intervals
use 3,000 paired site resamples, not independent-window or city sampling.

Holding each recipient and its actual blend coefficient fixed, the uniform
policy beats old-bank hindcast by **+0.418 pp** [+0.233, +0.660] and old-bank
minimum change by **+0.585 pp** [+0.403, +0.892]. These controls test the
forecast itself at an equal intervention count and coefficient; they do not
guarantee equal path displacement. The agreement cap retains slightly more
gain but passes fewer flight views. Neither arm passes the 2% outer
positive-easy calibration objective, despite low signed site averages.

## Why Recording Safety Still Fails

The original algebra requires selected known reference mass, selected
easy reference mass, and full easy reference mass to be defined per flight.
It checks the all-risk and easy-risk upper ratios (including the wholly
unknown-label debit) and the signed full-easy degradation. The denominator
and missingness checks were not changed. Under the OpenDD training-derived
easy cut, 30 of 606 flight-seed views have no full easy reference, equivalent
to 10 distinct flights; none can pass under any intervention policy on this
roster and cut. No identity exception is applied.

For the remaining views, a post-hoc outcome-aware test distinguishes poor
selection from infeasibility for the **frozen path per target**. A subset
can satisfy the original algebra if and only if at least one eligible
known-easy edit has harm no more than 2% of its CV error. Necessity follows
because the selected easy ratio is a reference-mass-weighted average of
individual harm ratios, plus a nonnegative missing-label debit. Sufficiency
follows by taking that single known easy edit: its all- and easy-risk ratios
coincide, and the full-easy signed degradation uses a denominator at least
as large. This statement is about feasibility, not positive utility.

| Uniform policy diagnosis | Flight-seed views |
|---|---:|
| Actual policy passes | 450 |
| Actual policy fails, but a qualifying singleton exists in hindsight | 112 |
| Full easy reference exists; every eligible easy edit exceeds 2% | 14 |
| No full easy reference | 30 |

This does not authorize a hindsight selector or redefining empty flights as
safe. It identifies two distinct gaps: some flights could benefit from a
better causal decision rule; others cannot demonstrate safety at all under
the current evidence and cut. No deployment change or new test access.
The next informative comparison is a causal flight-aware intervention rule
on development sources, while looking for independent sources with enough
reference support. Do not change the original safety rule, execute Stage5C,
or enable SMC.

```sh
# Requires the 21 hash-bound self-only checkpoints from the direct forecast run.
.venv-pytorch/bin/python scripts/run_m3w_opendd_velocity_calibration.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_velocity_flight_feasibility.py
```

The fresh run took 24.50 seconds on arm64 CPU with four compute threads and
no workers. Checkpoints and a PID heartbeat support resume; a separate
readout from saved checkpoints matched the result byte for byte in 2.58
seconds. No underlying tracks or checkpoints are distributed.

[Protocol](protocol.md) | [Full nested results](analysis.json) |
[Hindsight feasibility readout](flight_feasibility.json)
