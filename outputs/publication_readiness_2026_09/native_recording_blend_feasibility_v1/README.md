# Why Reweighting The Same Trajectories Does Not Fix Recording Risk

Fresh hindsight analysis of the 34 exposed development recordings,
three frozen seeds and 9,675 unchanged switch recipients. Each recipient
could take any affine mixture of the same two frozen paths. The original
CV floor, labels, masks, 2% limits and positive-support rule were unchanged.

| Recording-seed result | Views |
| --- | ---: |
| No selected positive easy reference mass; risk undefined | 23 |
| At least one minimum risk exceeds its allowance | 74 |
| Exact feasible witness re-scored under original rule | 5 |
| Unresolved at numerical tolerance | 0 |

All 74 numerical exclusions exceed the easy positive-harm limit; 66
also exceed the all-selected limit, and 27 exceed signed easy net
degradation. In 33 of the 74, the minimum unknown disagreement alone
is already larger than the entire easy allowance. These reasons overlap.
The 102 recording-seed views are not independent recordings.

For the five feasible views, the predeclared bounded ray yields positive
pair lower gain over the minimum-risk anchor in four. Four resulting
forecasts have positive conservative pair gain against CV, but **none
against history OLS4**. This search covers one ray; it is not an upper
bound on pair utility in those five views. Across all views, however,
no rule restricted to these exact recipients and these two trajectory
endpoints can pass the original recording risk contract. More training
of the same interpolator is therefore not the next useful experiment.
Changing recipients or adding an independently evaluated path direction
can escape this diagnostic; neither was tested here.

## Method

With the recipients fixed, the denominators are constants. The first two
numerators are sums of positive parts of observed ADE changes plus
maximum per-row disagreement with CV on wholly unlabelled rows. The
third sums signed ADE changes on easy rows. All are coordinatewise
nondecreasing in the same vector of scalar costs. Independently
minimizing the masked mean norm or the maximum norm in each row therefore
minimizes all three numerators together. A scalar subgradient bracket
and Lipschitz bound give a lower bound on each row minimum. The
impossibility calculation allows the identity path as a relaxation;
each feasible witness must preserve exact nonidentity and pass the
original evaluator. Undefined denominators remain failures.

[Protocol](protocol.md) | [Registration](registration.json) |
[Aggregate and recording-level evidence](analysis.json) |
[Exact replay receipt](replay_execution.json).

```sh
.venv-pytorch/bin/python scripts/audit_m3w_recording_blend_feasibility.py score
.venv-pytorch/bin/python scripts/audit_m3w_recording_blend_feasibility.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_recording_blend_feasibility.py \
  tests/test_m3w_pair_blend_direction.py tests/test_m3w_joint_path_audit.py \
  tests/test_m3w_dual_reference_capacity.py
```

13 focused tests passed. The local arm64 readout and replay each took
under four seconds after imports and produced the same result hash.
No CREATE job or external data role was needed. No new training,
independent confirmation, deployment, Stage5C execution or SMC.
