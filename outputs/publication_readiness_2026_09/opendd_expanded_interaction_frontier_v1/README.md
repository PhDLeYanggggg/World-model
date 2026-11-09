# How Much Interaction Gain Fits the Same Edit Budget?

**9 November 2026.** I held the forecast menus, exact edit counts and
displacements, and singleton decisions fixed. The question is whether
choosing different two-edit assignments can improve relative motion
without giving up accuracy or exceeding the original risk limits.
This extends the [accuracy-only upper bound](../opendd_exact_slot_headroom_v2/README.md):
the objective is now pair-relative prediction error, not individual
trajectory error. The data are the same 202 exposed OpenDD recordings
at three seeds. No model was trained or selected on this readout.

## Same Controls, Two Interaction Objectives

One solver choice is a full legal slate in a query. It fills every
frozen slot with a distinct recipient, leaves a CV recipient, and
does not change singleton forecasts. The constrained solver requires
the original three 2% risk checks, positive conservative individual
gain against both CV and OLS4, and no loss in conservative individual
gain against the original matched additive control. I reuse the
previous accuracy infeasibility certificates where applicable and
verify every new witness with the original risk scorer. Both menus
retain exactly **3,690 edits**. The larger menu adds only the existing
[held-source self-history forecast](../opendd_self_menu_v1/README.md).

For pairwise displacement, the observed score averages over query-local
co-observed steps. I also report a full-grid sensitivity: when a pair
target is absent, the decrease in its error is at least the negative
distance between the two forecasts' relative-displacement vectors.
This follows from the reverse triangle inequality. The lower gain
uses all requested pair steps in each multi-agent query; it cannot
become positive by omitting a difficult unlabelled step. The original
individual-forecast metric and risk tests are unchanged.

| Frozen menu | Observed: relaxed / constrained positive | Full grid: relaxed / constrained positive | Distinct recordings, constrained full grid |
|---|---:|---:|---:|
| OLS4 and median slope | 211 / 9 | 195 / 9 | 5 |
| Plus held-source neural path | 246 / 25 | 233 / 21 | 13 |

Each count is out of 439 edited recording-seed views. The 167 unedited
views remain separate, not risk passes. The unconstrained diagnostic
ignores risk and individual accuracy, not the budget. Of the 195
relaxed full-grid positives on the original menu, 117 already lack
enough individual gain to beat both references, 19 lack an easy
reference, and 47 are infeasible under the combined risk and baseline
constraints. Another three have no positive pair gain after imposing
the gates. The nine remaining views are at two locations. The expanded
menu's 21 are at three. All 174 new constrained solves returned
optimal, verified witnesses; there were no timeouts.

## What the Learned Policies Deliver

Each row compares the actual frozen joint policy against its own
matched additive counterpart. These are separate comparisons from
the ceiling, which keeps the original additive reference in both arms.
The location intervals use 3,000 paired draws after averaging seed
effects within recording and then recordings within location. Six
locations have matched edits. Positive values favor joint selection.

| Joint policy | Risk and both baseline gains, no accuracy loss vs additive | Also improves full-grid interaction | Full-grid gain (10^-3 native units), site mean [95% interval] |
|---|---:|---:|---:|
| Original | 32/439 | 3/439 | -0.225 [-0.623, -0.005] |
| Expanded menu, frozen scores | 31/439 | 2/439 | -0.422 [-0.922, -0.058] |
| Expanded menu, refitted scores | 31/439 | 0/439 | -0.031 [-0.067, -0.002] |

All three observed-only interaction intervals contain zero. A negative
full-grid **lower** gain is not evidence that the true unobserved
error increased. It means the gain is not robust to the missing
pair labels under this calculation. No evidence here establishes
physical safety or confirms a new scene result.

**Decision:** Do not scale the current joint head on this menu. The
larger menu expands the set of possible improvements but the tested
policies do not recover them reliably. For a new candidate menu,
measure whether it clears the individual accuracy and risk feasibility
checks before spending on a larger pair model. Freeze the resulting
method and compare against the same-budget additive policy on truly
independent, adequately labelled sources. This experiment measures
the feasible set itself, not a new ranking objective like
[Mandi et al. (2022)](https://proceedings.mlr.press/v162/mandi22a.html).

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/audit_m3w_opendd_interaction_frontier.py
.venv-pytorch/bin/python scripts/audit_m3w_opendd_expanded_interaction_frontier.py
.venv-pytorch/bin/python scripts/report_m3w_opendd_interaction_effects.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_interaction_frontier.py tests/test_m3w_exact_slot_headroom.py tests/test_m3w_pair_effect.py tests/test_m3w_slate_shared_menu.py tests/test_m3w_opendd_self_menu.py tests/test_m3w_exact_slot_costs.py
```

The solvers use arm64 local CPU and reuse only hash-bound parent
artifacts. Each fold writes an immutable checkpoint and PID heartbeat;
the same command resumes. No CREATE jobs, reserved outcomes, new
thresholds, manuscript edits or deployment changes.
[Base proof](../opendd_interaction_frontier_v1/readout.json) |
[Expanded proof](readout.json) | [Learned contrasts and attribution](realized_effects.json).
