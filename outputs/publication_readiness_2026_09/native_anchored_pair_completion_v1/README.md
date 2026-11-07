# Pair Bounds With Observed-Neighbor Anchors

I extended the [shared-pair check](../native_shared_pair_completion_v1/README.md)
to edges that share a missing agent but have different observed neighbors.
The two previous bounds reproduce on the same 34 exposed recordings,
14 sources, three seeds and 9,675 interventions per matched arm.

| Joint compared to | Previous lower (px) | Anchored lower (px), 95% CI | Extra tightening (px) |
| --- | ---: | ---: | ---: |
| Additive | -0.1569 | -0.1549 [-0.3340, -0.0353] | +0.0020 |
| Independent | -0.9264 | -0.9226 [-2.4304, -0.0409] | +0.0039 |
| Shared family | -1.1998 | -1.1917 [-2.7934, -0.1713] | +0.0081 |
| CV | -2.1145 | -2.1038 [-4.2534, -0.4993] | +0.0107 |
| OLS4 | -14.5562 | -14.3211 [-21.5077, -8.9430] | +0.2351 |

For joint versus additive the point envelope changes from
[-0.1569, +0.1738] to [-0.1549, +0.1718] px. It still crosses
zero. The paired 95% interval for the additional tightening
is [+0.000040, +0.005986] px, not an interval for the actual
unobserved effect. The conservative advantage is not established
for any of the five contrasts. An envelope spanning zero does not
show that both signs are actually attainable.

## What Changes In The Bound

If one agent is missing and the other is observed, write the
target relative vector as `missing_position - observed_position`.
Adding the observed position to each predicted relative vector
turns the original error into a distance to that same unknown
absolute position. Thus terms for different observed neighbors
can share one target by recording, missing agent and annotation
step. If both endpoints are missing, the older pair grouping
remains. We keep the less costly of the two valid recording
couplings, since the heuristic transport for unequal masses does
not promise optimality. No missing position is assigned.

There are 1,033,518 one-missing and 1,105,878 two-missing
pair-step occurrences across the repeated seed views. These sum
to the original 2,139,396 missing pair-steps. For the primary
contrast, 2,919 nonzero one-missing occurrences share a target
with another, and 185 single-target groups have a coupling
tighter than their diagonal bound. A count is not a separate
recording or an independent target across seeds.

This is sensitivity on pair ADE, not a new official endpoint
analysis, individual-utility correction or new 2% gate. It
keeps all observed labels, paths, sources and bootstrap draws.
No external labels, new training, deployment, Stage5C or SMC.

~~~sh
.venv-pytorch/bin/python scripts/audit_m3w_anchored_pair_completion.py score
.venv-pytorch/bin/python scripts/audit_m3w_anchored_pair_completion.py replay
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_anchored_pair_completion.py \
  tests/test_m3w_shared_pair_completion.py tests/test_m3w_joint_path_audit.py
~~~

Eight focused tests passed. Score and full replay each take about
33 seconds after imports on native arm64 and give the same result hash.
[Protocol](protocol.md) | [Aggregate](analysis.json) | [Replay](replay_execution.json).
