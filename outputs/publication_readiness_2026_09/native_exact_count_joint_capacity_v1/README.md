# Risk Feasibility Exists At The Same Exact Query Counts

The joint solver uses the same 9,675 total interventions and
the same count in every query as the three past-only recipient
rules. It can move the interventions to other eligible people.
The first hindsight arm keeps one path nomination per person,
identical to the operating comparison. The second can choose
from the full native bank. All results use the original 2%
all/easy, signed easy and positive reference checks.

| Label-aware candidate set | Checked feasible / 102 | Proved infeasible | Unresolved | Positive OLS4 witnesses |
|---|---:|---:|---:|---:|
| All recipients, fixed nomination | 93 | 9 | 0 | 44 |
| All recipients, full forecast bank | 96 | 6 | 0 | 51 |

Every checked feasible view has a proved optimum on the
conservative CV objective and strictly positive CV utility.
The OLS4 column only counts positive witnesses from that CV
optimization; it is not the OLS4-optimal ceiling. The full
bank may choose the least-error legal path on labeled rows
and the least-CV-disagreeing legal path on wholly unknown
rows without losing feasibility or CV utility. This reduction
does not claim dominance for the OLS4 objective on unknown rows.

Separate minima had only ruled out three views in each bank.
Solving all constraints simultaneously rules out nine or six.
Compare this with just **4/102** passes for the past-only
gain policy, **0/102** for support-first, and **2/102** for
risk shortfall, with exactly the same query counts and fixed
nomination set. The gap is not explained by the candidate set
or exact count alone. It motivates learning to recover feasible
recording decisions, not increasing the candidate bank by default.
It does not prove such decisions are identifiable from past data.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_exact_count_joint_capacity.py score
.venv-pytorch/bin/python scripts/audit_m3w_exact_count_joint_capacity.py replay
.venv-pytorch/bin/python scripts/plot_m3w_recipient_feasibility.py
~~~

The local solve took 179.4 seconds, with 12 refinements across
the two banks. Every feasible witness was rechecked with the
original scorer. The 1.5-second replay verified the existing
recording checkpoints and reproduced the aggregate exactly; it
did not repeat the solver work. Eight focused tests passed.
The vector figure binds the three original score artifacts and
keeps verified feasibility, unresolved bounds and actual passes
distinct. Figure SHA-256:
e78b7b840587832348b7cf907052d376cab7fd7b1e8772c77a584b05343285c7.

The 102 views repeat 34 exposed recordings at three forecaster
seeds. The diagnostic exports no label-aware decisions and adds
no independent result or deployment. Distances remain image
pixels on 8/12 native annotation steps, not meters or seconds.
No Stage5C execution or SMC.

[Protocol](protocol.md) | [Metrics](analysis.json) |
[Score](score_execution.json) | [Replay](replay_execution.json)
