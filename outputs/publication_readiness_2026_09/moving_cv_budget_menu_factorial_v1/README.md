# How much do the candidate menu and query budget limit forecasting?

I crossed candidate scope and edit counts on the same 24 exposed
MovingDroneCrowd++ development recordings, three seeds and nine
source groups. Every witness must pass the unchanged recording
risk/support checks against constant velocity (CV) and have
positive conservative utility over both CV and history OLS4.
The unit is a recording/seed view, not an independent trajectory.

| Candidate scope | Edits per query | Verified | Excluded | Unsupported | Unresolved |
|---|---|---:|---:|---:|---:|
| Narrow slate menu | Frozen exact counts | 5 | 51 | 16 | 0 |
| Full agent-path bank | Frozen exact counts | 19 | 32 | 15 | 6 |
| Narrow generators | Free prefix lengths | 5 | 54 | 6 | 7 |
| Full agent-path bank | Free counts | 53 | 13 | 6 | 0 |

The narrow policy family uses five causal geometric prefix rankings
and the original frozen slate. Its relaxed version can choose any
legal prefix length or no edits. The full bank can combine paths
across agents instead of inheriting a particular ranking. Two
narrow-relaxed witnesses came from the exact arm and were rescored
after a solver timeout; an unresolved solve cannot invalidate a
valid nested witness. Exclusions mean recorded infeasibility or a
negative conservative upper bound, not simply failure to find a plan.

Expanding the bank with counts fixed recovered 14 verified views
(source-balanced +12.17 pp, 95% paired bootstrap interval 0.00 to
24.34). Relaxing counts in the full bank recovered 34
(+43.39 pp, 17.46 to 70.37). With free counts, the full bank's gain
over the narrow family was +55.56 pp (25.93 to 85.19).
The narrow free-count arm recovered no additional verified view;
seven remain unresolved. Even giving those seven the benefit of
the doubt, 53 full-bank witnesses exceed the narrow upper count
of 12. These intervals describe the verified-witness rate,
not the true capacity of an unresolved arm.

The verified full/free solutions edit a median of 89.24% of
eligible rows in their recordings. This is **not** a minimum
necessary edit rate or a recommended deployment policy. Before
training another selector, the actionable next question is which
of those views retain capacity at an intermediate budget, with
a wider shortlist that can be constructed causally.

## What this adds to the paper

[Decision-focused learning to rank](https://proceedings.mlr.press/v162/mandi22a.html)
addresses scoring feasible combinatorial solutions. This comparison
holds the forecasts and risk rule fixed and measures how often the
candidate construction prevents even an outcome-aware ranker from
having a valid solution. It does not establish a new ranking method.
[Joint Metrics Matter](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html)
motivates evaluating compatible multi-agent futures. The feasibility
counts here use the original recording risk and dual accuracy
criteria, not collision metrics or a claim of joint physical safety.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/audit_m3w_cv_budget_menu_factorial.py --replay
.venv-pytorch/bin/python scripts/verify_m3w_cv_budget_menu_inclusion.py
.venv-pytorch/bin/python scripts/build_m3w_cv_budget_menu_replay.py
```

The middle arms are newly solved; the narrow/exact corner is
recomputed using the original solver and the full/free corner's
prior certificates and witnesses are verified. The first command
checks saved local witnesses; omit `--replay` to resume or solve.
The NumPy-only anonymous archive reproduces the scalar results and
3,000 source-group bootstrap draws, not training or the solver.
This is development-only, outcome-aware diagnosis. No policy
was trained or deployed and no DroneCrowd confirmation was opened.
The task is native 8 observed / 12 future steps in dataset-local
pixels; this does not show metric accuracy or physical safety.

[Protocol](protocol.md) | [Initial solver readout](readout.json) |
[Verified comparison](nested_verification.json) |
[Manuscript v134](../evidence_manuscript_v134/README.md)
