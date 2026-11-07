# Which Recipients Are Missing At A Fixed Budget?

I expanded three things separately: how existing paths can be combined,
which paths the same people may receive, and finally who may receive a
path. The intervention cap stays at the original matched count for
each query. The actual number used may be lower.

| What the hindsight solver can use | Risk-feasible views | Positive conservative gain vs. OLS4 | Proved infeasible |
| --- | ---: | ---: | ---: |
| Prior twelve intact query options | 71 | 33 | 31 |
| Recombine the existing recipient-path pairs | 75 | 34 | 27 |
| Full forecast bank, same recipients | 77 | 39 | 25 |
| Same full bank, all query recipients | 99 | 51 | 3 |

The last expansion adds 22 feasible views and 12 positive views across
five sources. It also removes every case with no eligible easy reference
candidate; the three remaining infeasible views do have an easy candidate
but cannot jointly satisfy the risk constraints. Among 77 views feasible
before the recipient expansion, the median increase in optimum lower
utility is 3.711 percentage points on the unchanged CV denominator.
Even after expanding recipients, **48 feasible views have a nonpositive
OLS4 upper bound on the conservative objective**.

This is a finite action-space diagnostic, not a learner or a new
intervention budget. It uses exposed labels from 34 Moving A recordings
in 14 sources, repeated at three seeds (102 dependent views), with
eight observed and twelve predicted annotation steps in image pixels.
CV is the fallback and reference for the original 2% risk and positive
easy-support checks. OLS4 is the accuracy comparator. Neither absolute
physical safety nor an independent improvement follows from this result.

Nine full-bank solves initially timed out. I kept the first result,
then reran only those nine with a longer cap and exact dominance
pruning. An unobserved action with no incremental lower gain is
dominated by abstention; one with higher disagreement from CV and no
better disagreement from OLS4 is dominated by its same-recipient
alternative. The extended solves all finish, and every feasible witness
is checked with the original scorer. No selected recipients or
forecast arrays are published.

This prioritizes broader past-only retrieval over another trajectory
variant restricted to the old recipients. The gap to reliable learned
retrieval still needs a held-source, same-budget comparison. DroneCrowd
calibration and the confirmation split are unchanged. No deployment,
Stage5C or SMC.

[Protocol](protocol.md) | [Initial result](analysis.json) |
[Refined result](refinement.json) | [Run receipt](refinement_execution.json)

```sh
.venv-pytorch/bin/python scripts/audit_m3w_recipient_budget_capacity.py score
.venv-pytorch/bin/python scripts/refine_m3w_recipient_budget_capacity.py score
```

Run from the root with the existing hash-bound Moving A local assets.
The first command checkpoints by recording and seed; the second resumes
only the original unresolved all-recipient views. Each command verifies
the same aggregate on a repeat invocation with the local checkpoints.
