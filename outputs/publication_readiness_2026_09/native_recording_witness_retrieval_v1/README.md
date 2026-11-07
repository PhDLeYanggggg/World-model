# Learning From Recording-Level Witnesses

I replaced the local opportunity label with membership in a
recording-level solution that satisfies the original risk rule and
achieves positive conservative gain over OLS4. A witness is computed
from exposed labels only, permits changes on labeled rows, and still
charges the CV fallback's disagreement from OLS4 on wholly missing
rows. The restricted teacher finds 51 positive views and labels
5,274 recipient instances; 99 solves are optimal and three infeasible.
For each retriever fit, the entire scored source is excluded.

Both arms use the same past-only features, forecast bank, first four
gain-ranked slots, eight recipients in total, identical original query
caps and the same decoder and gate. There are 42 deterministic
source-excluded fits over three frozen forecaster seeds. Each arm
uses 38,688 shortlist slots and proposes 9,675 switches before gating.

| Ranker | Risk-feasible capacity / 102 | Positive OLS4 capacity / 102 | Actual gated passes / 102 |
| --- | ---: | ---: | ---: |
| Gain only | 81 | 45 | 0 |
| Recording witness | 87 | 48 | 4 |
| Original unshortlisted controller | Not evaluated here | Not evaluated here | 3 |

The new shortlist gains six feasible and three positive views, without
losing any. The paired source-weighted positive contrast is +3.571 pp
[0.000, +10.714]. This is not a reliable improvement in accuracy:
at matched counts, conservative utility is -0.361 pp
[-0.828, -0.023] versus gain-only. After gating, the witness arm
switches 72 times (original: 90), has 88 undefined easy views, and is
-9.819 pp [-21.998, -0.549] versus OLS4. The extra passing view does
not establish a deployable improvement.

## Where The Opportunity Is Lost

The separate [retention diagnostic](retention.json) allows a hindsight
solver to abstain at each query, under the same risk and utility
definitions. It successively restricts which paths the solver may use:

| Options retained | Risk-feasible views | Positive OLS4 views |
| --- | ---: | ---: |
| All legal paths on the retrieved recipients | 87 | 48 |
| One intact decoded option per query | 62 | 26 |
| Only decoded options admitted by the fixed gate | 7 | 3 |

All 102 views have resolved feasibility and objective status at every
stage. The loss occurs both when selecting one path assignment and
when filtering that assignment. Keeping more than one legal path
assignment for the causal decision stage is a more direct next test
than re-ranking the same shortlist again. The table does not show that
a past-only policy can find the hindsight choices.

This is exposed development: 34 recordings, 14 sources, three seeds,
3,000 paired source draws after seed averaging, eight observed and
twelve predicted annotation steps in pixels. The original controller
and 2% rule remain unchanged. No independent input or outcome,
deployment, Stage5C or SMC.

[Protocol](protocol.md) | [Accuracy and risk](analysis.json) |
[Retention registration](retention_registration.json)

```sh
.venv-pytorch/bin/python scripts/run_m3w_recording_witness_retrieval.py fit
.venv-pytorch/bin/python scripts/run_m3w_recording_witness_retrieval.py score
.venv-pytorch/bin/python scripts/run_m3w_recording_witness_retrieval.py replay
.venv-pytorch/bin/python scripts/audit_m3w_recording_witness_retention.py score
```

Teacher memberships and frozen forecasts stay in the private local
cache. Repeated scoring reconstructs operating metrics from the
frozen decisions and verifies the light aggregate.
