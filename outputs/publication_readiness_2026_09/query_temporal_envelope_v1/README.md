# Learning when a forecast edit should take effect

**Decision: no policy change.** I tested whether a neural model could
improve the timing of an edit without changing who receives it, its
direction at any step, or its maximum departure from constant velocity.
This is a fresh development experiment on MovingDroneCrowd++, not a
rerun of the official DroneCrowd calibration or confirmation.

## Matched test

The verified parent nominates the same recipients and candidate directions.
Every arm uses the exact same 95,769 edits across 72 recording-seed views.
For each selected row, every shape has a peak displacement equal to one
quarter of the original candidate's peak; equal peak does not imply equal
total distance across time. The network learns four positive knots over
12 forecast steps from causal descriptors, starting at the flat envelope.
It trains on normalized available-step trajectory error and excess error
over CV and OLS4, with extra weight on easy CV harm. Neither target
coordinates nor future masks enter the network. All 27 fits use 500
updates in nine source-separated folds and seeds 17, 29 and 43. The
six fitting, two inner calibration and one outer scoring groups are
disjoint within each fold. The scorer and original 2% rule are unchanged.

| Envelope | Conservative gain vs CV (pp) | All-risk violations / 72 | Risk passes / 72 |
|---|---:|---:|---:|
| Flat quarter | +1.178 | 45 | 0 |
| Earlier ramp | +1.817 | 57 | 0 |
| Later ramp | +0.715 | 30 | 0 |
| Learned | +1.264 | 45 | 0 |

The primary **direct paired** learned-minus-flat contrast is -0.016 pp
(95% CI [-0.425, +0.389]); learned minus the earlier ramp is -0.751 pp
([-1.880, -0.042]). The learned arm also loses against OLS4:
-10.090 CV-normalized pp [-18.476, -1.673]. The intervals use 3,000
paired source-group draws after averaging the actual training seeds.
No static or learned option qualifies in any inner fold-seed; the
calibrated arms both fall back to CV and do not establish safety or lift.

## What timing alone can fix

A separate, label-aware diagnostic relaxes the learned four-knot envelope
to independent nonnegative motion on the same ray at each step. Project
each observed target onto the ray segment with the fixed peak as upper
bound. To preserve the exact peak, force one nonzero direction step to
that peak, choosing the smallest increase in observed error. An unobserved
step with a nonzero direction costs nothing in this optimistic relaxation.
This minimizes available-step ADE for each selected known row and therefore
simultaneously minimizes the three original risk numerators: the selected
reference masses do not change and wholly unknown rows have unchanged
peak debit. It is not a policy, a physical safety claim, or an upper bound
for different recipients, directions or peak budgets.

Only **4/24** distinct recordings can pass even this relaxed timing
ceiling. Five have undefined easy risk; of the remaining failed recordings,
two first fail all-risk and thirteen first fail easy risk. The oracle's
CV-relative lower utility is +4.459 pp [1.738, 7.755], but it requires
future labels. No more tuning of the four knots can resolve the fixed
allocation's recording-level gate.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_query_temporal_envelope.py
.venv-pytorch/bin/python scripts/run_m3w_query_temporal_envelope.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_query_temporal_capacity.py
```

Native arm64 Torch, four CPU threads, no loader workers; fresh run
29.91 s, checkpoint-only replay 21.93 s. The replay reproduced the
bound calibration and result files without a refit. Checkpoints include
optimizer and sampler state every 100 updates; rerun the same command
to resume. 33 targeted tests passed. No official calibration or
confirmation access, deployment change, Stage5C or SMC.

[Registration](registration.json) | [Training and evaluation](analysis.json) |
[Hindsight timing capacity](capacity.json).
