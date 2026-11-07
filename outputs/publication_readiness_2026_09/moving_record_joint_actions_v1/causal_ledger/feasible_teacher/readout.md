# An Outcome-Optimal Teacher Is A Degenerate Risk Control

I compared demonstrations that maximize known conservative utility with
demonstrations that impose the original all/easy risk excess constraints.
Both use identical current-query counts. In all 4,825 active query views
(1,608 / 1,610 / 1,607 across seeds 17/29/43), the two teachers choose
exactly the same actions. Fourteen repeated query views provide no
demonstration. The unrestricted teacher already sees the true outcomes,
and its best gains rarely require selecting a harmful path.

The 84 source-excluded student fits inherit identical training targets.
Their held-source choices and outcomes are identical, including 6,184
switches and four of 102 original risk passes for each student. The
frozen ledger passes three. A numerical zero contrast between identical
arms does not identify the effect of constrained teaching. In particular,
four passes are not evidence that constraints improved safety.

I therefore ran the [nested proposal repair](nested/readout.md): replace
the label-aware objective with causal utility from fits that exclude
both the evaluated source and the current demonstration donor. Observe
the donor outcomes only to screen the risk of these proposals.

Original 2% rules and support handling unchanged. All views are already
exposed development data, not independent confirmation. No deployment,
Stage5C or SMC.

```sh
.venv-pytorch/bin/python scripts/run_m3w_feasible_teacher.py fit
.venv-pytorch/bin/python scripts/run_m3w_feasible_teacher.py score
.venv-pytorch/bin/python scripts/run_m3w_feasible_teacher.py replay
```

Registered at `7872d572`. Frozen before evaluation. Exact score replay.
[Protocol](protocol.md) | [Replay receipt](replay_execution.json)
