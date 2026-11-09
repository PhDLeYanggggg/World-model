# Can a ranker learn from full-record feasible assignments?

I changed who can receive an edit and which path that person receives,
while keeping the controller's 1,457 intervention slots at exactly the
same queries and peak displacements. A recipient must have enough
native CV-to-OLS4 displacement and cannot take two slots at a query.
The menu contains the past-prefix analytic shape and the comoving
spline at each eligible recipient and radius. The old analytic plan
is always available.

I fit two matched rankers. The gain teacher chooses a full-record plan
to maximize the smaller conservative gain against CV and OLS4. The
risk teacher uses the same objective, with the original recording
risk and easy-reference requirements. If no positive witness is
found, the teacher returns the original assignment. Both students
use the same pairwise imitation loss, current history and candidate
geometry. The query decoder enforces distinct recipients and slot
budgets, not the label-dependent recording rule. This is related to
[optimization-based ranking](https://proceedings.mlr.press/v162/mandi22a.html),
but no gradient passes through the teacher solver and no risk
guarantee is inherited from its solutions.

I trained 42 rankers for 25,200 optimizer updates across seven locations
and three seeds. The donor slot generator and spline exclude the donor
and outer location. The held predictions are frozen before scoring. The
1,635 qualified constrained-teacher training views repeat donors
across folds and seeds; they are not independent examples.

| Policy | Recipient changes | Spline uses | Easy undefined | Risk passes | Risk and both gains |
| --- | ---: | ---: | ---: | ---: | ---: |
| Analytic anchor | 0 | 0 | 334 | 112 | 77 |
| Past-prefix substitution | 0 | 0 | 334 | 124 | 90 |
| Gain teacher student | 753 | 885 | 361 | 104 | 77 |
| Risk teacher student | 260 | 289 | 345 | 109 | 76 |

The risk student gains five risk passes over the gain student, but
loses one dual pass and **0.025 conservative pp** of accuracy
(95% paired location bootstrap **[-0.048, -0.002]**). Against past
prefix it loses **0.019 pp [-0.037, -0.003]** and 14 dual passes.
Worst signed easy degradation is 11.28% for the risk student and the
past-prefix control, versus 15.13% for the gain student. Counts are out
of 606 views from 202 recordings; intervals average seeds before
3,000 paired resamples of seven locations. Neither student passes
the original recording rule across the roster.

## Where the gains disappear

With outcomes known, the same-query, same-radius analytic menu
qualifies **206** views. The wider menu with spline choices qualifies
**284**, with no unresolved solves. Two locations supply 76 of the 78
additional views. If the risk student's recipients and their radii are
locked, re-optimizing shapes with outcomes qualifies only **168** views.
This is even below the **170** reachable at the original recipients.
The trained model therefore did not discover better recipients overall;
the missing improvement cannot be recovered simply by changing the
shape selector after recipient choice. The gap between 76 and 168
also shows that shape choice at the learned recipients still matters.

Next I would target support loss directly, by contrasting candidate
recipients against the actual original assignment instead of imitating
one optimum for the full recording. The strong location concentration
also calls for more varied fitting sources rather than more seeds on
the same records. No deployment change, independent confirmation,
Stage5C or SMC. These are native 8/12 annotation steps and publisher
coordinates; no seconds, metric or physical-safety claim.

```sh
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/run_m3w_opendd_feasible_recipient_imitation.py --replay
OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_feasible_recipient_imitation.py
```

Omit `--replay` to fit or resume checkpoints. Replaying produced the
same predictions and readout with zero optimizer updates. The capacity
and attribution calculations use outcomes for analysis, not inference.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Readout](readout.json) |
[Attribution](attribution.json) | [Manuscript v152](../evidence_manuscript_v152/README.md)
