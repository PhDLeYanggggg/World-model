# Can A Past-Only Ranker Imitate Feasible Joint Decisions?

**Fresh development experiment; not independent confirmation.**
The exact-count audit showed that useful candidate assignments exist
in hindsight. This experiment asks whether those assignments teach a
ranker to find good replacements from past observations alone.

The [registered protocol](protocol.md) uses a CV-reference, label-aware
optimizer for each training recording only. It solves for the original
per-query counts and recording risk and support rules. In the full
forecast bank it verifies 96 teachers and proves 6 views infeasible
(32 feasible recordings and 1,594 usable query views per seed). For
each of 14 excluded source groups and three frozen forecasting seeds,
a pairwise ridge ranker learns from the other sources. It compares the
teacher's choice with other recipients and with other paths for the
same person. The student uses past features and fixed path-family
indicators; the future labels enter teacher construction and scoring,
not held-source prediction. This is imitation of one valid assignment
per recording, not learning the unique optimal assignment.

| Frozen choice | Matched switches | Recording-seed views passing | Conservative utility vs gain (pp, 95% CI) |
| --- | ---: | ---: | ---: |
| Gain ranking | 9,675 | 4/102 | Reference |
| Support-first ranking | 9,675 | 0/102 | See prior experiment |
| Risk-shortfall ranking | 9,675 | 2/102 | See prior experiment |
| Student, fixed path nomination | 9,675 | 0/102 | -2.020 [-4.469, -0.449] |
| Student, full action bank | 9,675 | 0/102 | -16.607 [-25.703, -9.080] |

Intervals resample 14 source groups 3,000 times after averaging over
seeds. In the full-bank comparison, the available-label contrast is
-1.501 pp [-3.865, +0.353]. The 15.106 pp difference from the
conservative lower contrast is a debit for forecast disagreement on
wholly unobserved rows, not a measurement of actual error there. The
observed relative-path ADE reduction is -1.582 image pixels
[-4.491, +0.466]. The worst signed easy degradation is 98.702% for
gain, 131.137% for the restricted student and 1025.283% for the full
student. Applying the registered screen retains zero full-student
switches; all 102 selected easy ratios are undefined. This is neither
a safety pass nor a matched utility comparison.

In a descriptive join of the frozen choices against the private
teacher selections by recording, seed and row, 9,564 teacher
selections matched 464 recipients under gain, 1,318 under restricted
imitation and 1,158 under full imitation (140/621/493 exact actions).
These counts measure agreement with a hindsight witness, not unique
pedestrians or the ability to meet recording constraints. Better
imitation does not ensure that the held recording's selected support
and harm work out jointly. The experiment does not identify which
information is impossible to infer from the available histories.

The protocol uses eight observed and twelve predicted annotation
steps in native image pixels. This is not a metric or seconds-based
test. The source-held development set is already exposed. No model
promotion, external access, Stage5C execution or SMC.

```sh
for phase in fit freeze score replay; do
  .venv-pytorch/bin/python scripts/run_m3w_structured_imitation.py "$phase"
done
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_structured_imitation.py \
  tests/test_m3w_exact_count_joint_capacity.py tests/test_m3w_recipient_shortfall.py \
  tests/test_m3w_fixed_recipient_risk_bound.py
```

Local fitted artifacts and decision arrays are excluded from Git.
The [fit](fit_execution.json), [freeze](freeze_execution.json),
[score](score_execution.json) and [replay](replay_execution.json)
receipts bind the outputs. Replay matches [the scalar analysis](analysis.json)
exactly; all seven focused tests passed.
