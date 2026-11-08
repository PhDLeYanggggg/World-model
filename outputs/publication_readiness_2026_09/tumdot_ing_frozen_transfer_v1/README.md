# Frozen OpenDD-to-TUMDOT-ING Transfer

**Result: no transfer gain over CV or matched motion, and no safety pass.**
The 5% neural blend loses 0.242 percentage points to CV and 0.795 points
to past-only hindcast under the missing-step lower comparison. Both
contrasts are negative in each of the nine recording means and on each
day. Against the weaker OLS4 reference the same blend gains 4.039 points;
that comparison alone would misrepresent the transfer result.

The [registered protocol](protocol.md) fixes the file roster, coefficients
and readout. From [the TUM release](https://www.mos.ed.tum.de/en/vt/forschung/datensaetze/tumdot-ing/)
I used the first listed raw file per intersection-day in the three-day
Ingolstadt campaign. All nine matched the publisher's catalog sizes. Three
files additionally match the prior structural-audit hashes. The nine time
axes fit the same 29.97 Hz grid without gaps and the local coordinates
are translations of the released UTM coordinates. These statements apply
to these release files, not SDD or DroneCrowd. The sample is three nearby
intersections in one campaign, not nine independent scenes. No interval
estimates across campaigns are possible here.

The causal 8-to-12 roster contains 404 non-overlapping queries within
recordings, 1,456 target rows, 1,259 complete futures and 17 wholly
unobserved futures. Each seed view averages the same seven hash-verified
OpenDD producers. This was fresh inference and scoring, not new model
training. No TUMDOT fit, calibration, or threshold search was performed.

| Arm | Lower gain vs CV (pp) | vs OLS4 (pp) | vs matched hindcast (pp) | Original risk passes |
| --- | ---: | ---: | ---: | ---: |
| 5% neural blend | -0.242 | +4.039 | -0.795 | 19/27 |
| 25% neural blend | -1.611 | +4.124 | -3.469 | 0/27 |
| Direct neural forecast | -11.740 | -2.517 | Not matched | 0/27 |
| OLS4 | -15.861 | 0.000 | Not matched | 0/27 |

Equal-file averages over three existing seeds; units are percentages of
observed CV error mass, not population estimates. Missing-step comparisons
are pair-specific lower bounds and cannot be subtracted. Matching fixes the
recipients and blend coefficient, not the resulting displacement magnitude.
No editing for CV leaves selected risk undefined; it is not counted as
a certification under the original rule.

## Why It Failed

The [post-hoc decomposition](decomposition.json) leaves the selected policy
unchanged. The 5% blend gains just +0.040 points on observed steps against
CV and +0.033 on complete rows. The 0.282-point missing-step debit makes
its conservative CV contrast negative. Against matched hindcast, it is
already worse on observed steps (-0.335) and complete rows (-0.309);
missing labels alone cannot explain that deficit. All eight failed neural
views exceed the 2% upper easy-risk limit, despite a worst signed net
easy change of only +0.351%. The matched motion control is +0.008 points
conservatively versus CV but itself passes only 12/27 risk views. Neither
alternative is safe for deployment on this readout.

The result narrows the modeling problem: improve forecast quality relative
to matched causal motion on complete donor histories, and address positive
harm separately from average signed error. Do not increase the blend
coefficient or tune it on this readout.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_tumdot_ing_frozen_transfer.py register
.venv-pytorch/bin/python scripts/run_m3w_tumdot_ing_frozen_transfer.py run
.venv-pytorch/bin/python scripts/run_m3w_tumdot_ing_frozen_transfer.py replay
.venv-pytorch/bin/python scripts/diagnose_m3w_tumdot_ing_frozen_transfer.py
```

Input file identities, donor cohorts and code are bound in
[registration](registration.json); the [analysis](analysis.json) and
[exact replay](replay_execution.json) contain recording and seed results.
Both executions used native arm64 CPU and four Torch threads. The targeted
causal-roster, missing-label and admission tests passed (16 tests).

The 15/30 DroneCrowd split is unchanged. Its calibration found no
qualified threshold; its 30 confirmation groups were not accessed.
No formal confirmation, deployment promotion, metric or seconds claim for
SDD, true 3D, foundation model claim, Stage5C execution, or SMC.
