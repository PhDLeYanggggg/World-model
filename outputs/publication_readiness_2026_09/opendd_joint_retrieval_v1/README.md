# OpenDD: Joint Retrieval Recovers Options, Not Forecast Gains

The conditional pair head could not choose useful slates that had
already been dropped from the feasibility shortlist. I changed
retrieval while keeping the forecast menu, frozen weights,
additive reference, final ranking rule and shortlist size fixed.

## Matched Comparison

The original shortlist keeps the eight highest feasibility scores.
The new shortlist retains the four highest, then fills up to
eight using the average of the feasibility and predicted
conservative pair-gain ranks. Duplicate indices are excluded.
Both choose the highest positive predicted full-grid relative
gain among their eight, otherwise the highest-feasibility
candidate. The new arm scores pair gain over the full legal
menu for retrieval; only the final shortlist budget is matched,
not the retrieval computation.

I replayed 21 location-seed folds and reproduced the original
shortlist forecasts and probabilities exactly. Training,
calibration and held roles did not change. The forced comparison
uses the same 966 exact edit slots across 159 edited
recording-seed views. Optional release uses the same calibration
rule and original 2% checks. No model was refit.

## Retrieval Versus Realized Utility

| View set | Any locally qualifying pair in full menu | Retained in original / joint shortlist | Selected by original / joint shortlist |
|---|---:|---:|---:|
| Calibration | 141 | 42 / 56 | 22 / 20 |
| Held locations | 150 | 40 / 51 | 20 / 22 |

On held locations the joint shortlist retrieves 17 new
opportunities and loses six. Its net increase in shortlist recall
is 7.33 pp, but it selects only two more qualifying local slates.
The latter are query-seed views, not independent recordings or
release passes. The full-menu scorer processes 192,801 candidate
scores across 483 held queries (including seed repeats) and
changes 330 forecasts relative to the original shortlist.

| Forced policy | Known ADE | Available FDE | Mean recording p95 ADE | All risk checks | Easy over 2% | Undefined edited-easy |
|---|---:|---:|---:|---:|---:|---:|
| Feasibility shortlist | 0.54772 | 1.25408 | 1.39120 | 15/159 | 49 | 52 |
| Joint shortlist | 0.54767 | 1.25386 | 1.39108 | 13/159 | 43 | 51 |

CV remains the strongest tested causal reference (ADE 0.54726,
FDE 1.25329). The joint shortlist's conservative CV gain is
−0.0537 pp [−0.1517, +0.0416], and its hard gain is −0.026%.
Qualifying positive full-grid pair views versus additive decline
from eight to three. The **direct** paired conservative trajectory
gain versus the old shortlist is −0.7242 pp [−2.2609, +0.1233].
Neither method yields a nonempty calibrated policy: 0/483 score
breakpoints each. Both fall back to CV; the empty comparison is
not a risk pass.

## Full-Grid Attribution

The direct full-grid relative gain, joint minus original shortlist,
is **−0.004605 native units** with paired location 95% interval
**[−0.013860, +0.001560]**. On the *same full-grid denominator*
the contribution from known relative positions is **+0.000031**
and the missing-step debit is **0.004636**. The sum reproduces
the direct lower gain. rdb5 has the largest missing-step debit
(0.015888); location intervals use 3,000 draws over four estimable
locations after averaging seeds within recording.

Improving shortlist recall is therefore insufficient. The extra
choices do not deliver a detectable gain on known steps, and
they change relative predictions in places with no verified
future pairs. For the next model, first test a fixed budget of
alternatives with smaller relative displacement from the additive
reference or better past-only evidence for pair observability.
Keep the existing missing-step debit and risk rules. This
development result does not qualify a policy for independent
confirmation or deployment.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_joint_retrieval.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_joint_retrieval.py score
.venv-pytorch/bin/python scripts/audit_m3w_joint_retrieval_debit.py
```

The native arm64 freeze took 41.84 seconds and scoring 38.50
seconds, under 773 MB maximum RSS. Four Torch threads, no loader
workers; each fold has atomic forecasts and a phase/PID heartbeat.
Nineteen related tests passed. No CREATE job, Stage5C, SMC or
reserved outcome access.
[Registration](registration.json), [metrics](summary.json),
[full-grid attribution](debit_audit.json).
