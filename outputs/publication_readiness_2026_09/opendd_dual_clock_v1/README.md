# Learning When to Spend Both Intervention Slots

I let both the gain and support edits move rather than changing
only direction at fixed rows. The same frozen neural bank provides
the two amplitudes, 0.25 and 0.10. At each query the gain and easy
heads nominate up to two rows per role. The online policy sees
only current candidates, current query index and past committed
actions. It cannot see later queries or force an edit at the end.

I trained the head on current value, best value if acting now
and completing within the next two queries, and the latter value
minus waiting. Donor labels use conservative gain against CV and
OLS4 and the unchanged whole-record risk rule; no-edit receives
zero *decision* value, not a risk pass. Fitting forecasts and easy
heads exclude both the donor and the evaluation site. The donor
gain heads reuse the separately frozen, triple-exclusion lineage.
No model selection or threshold tuning uses the evaluation scores.

The matched earliest control knows only which roles the online
arm ultimately spends in each recording. It chooses the earliest
valid schedule in that same nominee bank. It is retrospective,
uses no outcome labels and is not deployable. The contrast holds
role count and amplitude fixed per recording, not exact radius
or recipient; these can change when an edit moves.

| Policy | Edits | Undefined | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: |
| Frozen independent slots | 943 | 295 | 172 | 138 |
| Myopic learned score | 150 | 542 | 20 | 16 |
| Myopic, matched earliest | 150 | 545 | 34 | 30 |
| Learned continuation | 146 | 542 | 20 | 16 |
| Continuation, matched earliest | 146 | 546 | 34 | 30 |
| Learned continuation with fixed deadline | 581 | 407 | 113 | 75 |
| Deadline, matched earliest | 581 | 428 | 96 | 81 |

The primary continuation-minus-matched conservative contrast is
**-0.00626 pp [-0.01925, +0.00048]**. It moves 88 of 146 edits
relative to the matched control but loses 14 joint passes.
The 146-edit arm does not reliably improve on CV.
At the old per-recording role counts, an exact search over these
same nominated rows verifies **428/606** feasible views. All legal
rows, at the same counts and amplitudes, give 473 in the
[previous audit](../opendd_support_schedule_v1/README.md).
Thus nomination explains some loss of capacity, but not the
learned schedule's failure to find the feasible actions.

The fixed deadline is a post-result development diagnostic with
no refitting. Each role releases at its first qualifying query;
if still unused two queries later, it is used only if a nominated
row is present, otherwise it expires. There is no end-of-record
release. It restores substantial coverage and gives +0.08070 pp
[+0.01686, +0.15074] versus CV, but the matched contrast is
**-0.03732 pp [-0.15095, +0.05615]** and joint passes still fall
relative to the same-budget control. Improving coverage is
not sufficient to learn when the edits help.

Intervals average the three seeds and resample seven sites 3,000
times. The 606 views are 202 previously exposed OpenDD recordings,
not an independent experiment. Neither arm passes the original
full risk rule. The next test should separate budget allocation
from ranking in the learning objective and assess the ranking
against a matched control, rather than lower the abstention
threshold on the same exposed sites.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_dual_clock.py
.venv-pytorch/bin/python scripts/replay_m3w_opendd_dual_clock.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_dual_deadline.py
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_calibration_roles.py verify
```

The first command fitted 21 heads (12,600 optimizer updates,
arm64 Torch, four threads and no workers), wrote all checkpoints,
and then hit a NumPy counter serialization error. The recovery
script reads those checkpoints and converts diagnostic counts
to native integers before aggregation. The first attempted
conversion lost the Counter type; v2 preserves it and finished
with zero optimizer updates. No policy or threshold changed.

`fresh_run`: training, exact nomination capacity and the deadline
diagnostic. `cached_verified`: nested trajectory and nomination
lineage, completed checkpoints. `not_run`: DroneCrowd independent
confirmation. The 15/30 roles and 2% rule are unchanged; no new
deployment, Stage5C, SMC, CREATE job or paper submission.
