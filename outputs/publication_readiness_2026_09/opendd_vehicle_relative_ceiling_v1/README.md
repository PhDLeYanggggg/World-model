# OpenDD: exact interaction headroom at the same two edits

Beating two motion references is not the same as improving interaction
over an already matched additive decision. I enumerated 3,691,167 legal
pairs on the unchanged vehicle menu. Each pair edits two distinct actors
at the same query and peak displacement. The study has 493 unique
recordings, seven locations, and 8,874 repeated recording views over
three seeds and six calibration directions.

| Question about the same candidate menu | Views / 8,874 |
| --- | ---: |
| Any pair passes the original 2% checks and beats CV and OLS4 | 8,062 |
| Also no conservative trajectory loss vs the matched additive output | 5,611 |
| Also positive conservative relative motion vs that output | 1,708 |
| A same-family pair satisfies all of the above | 1,640 |
| A pair retaining at least one additive recipient satisfies all of the above | 1,632 |
| Additive itself also qualifies, and a joint improvement exists | 716 |
| Unrestricted learned joint actually satisfies the full relative-motion criterion | 242 |

Of the 1,708 hindsight opportunities, only 68 require unrestricted
families. The frozen learned joint chooses a qualifying pair in 242
views, about 14.2% of the broad hindsight count. That ratio is not the
capture rate on the stricter 716-view subset. Under equal-location
weighting the broad opportunity rate is **17.18%** [12.51%, 21.70%],
and realized joint success is **2.42%** [1.98%, 3.01%]. Intervals
resample seven location means 3,000 times, not individual views.

This explains why a high dual-reference oracle ceiling is insufficient
evidence for joint intervention. A smaller but nonzero opportunity
remains under the stronger matched criterion. Most of it does not
require adding families or replacing both recipients. The most
targeted follow-up is to test a restricted, additive-anchored
ranker, not to enlarge the candidate menu based on the old dual
ceiling. A hindsight witness does not show that the needed
comparison is predictable from eight observed steps.

## Exact edge calculation

For an actor pair and step, cost is the relative-position error if
both labels are available, and the relative forecast disagreement
against the additive output otherwise. Start from the all-CV cost.
Each single edit changes just the edges incident to its actor.
Adding the two changes counts their shared edge incorrectly; replace
the two single-edit changes on that edge with its joint change.
This gives exactly the same lower contrast as constructing the
full joint forecast for every candidate. With N actors, M legal
single edits, P candidate pairs and T steps, the edge work is
O(T(N^2 + MN + P)) instead of O(TPN^2).

Every selected witness replays with the original risk, reference
and pair scorers. The old 8,062-view dual capacity and both actual
control counts reproduce. This is an oracle for a bounded decision
on fixed forecasts, not a joint trajectory generator such as
[FJMP](https://openaccess.thecvf.com/content/CVPR2023/html/Rowe_FJMP_Factorized_Joint_Multi-Agent_Motion_Prediction_Over_Learned_Directed_Acyclic_Interaction_Graphs_CVPR_2023_paper.html)
or end-to-end decision-focused training.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/diagnose_m3w_vehicle_relative_ceiling.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_relative_ceiling.py tests/test_m3w_vehicle_pair_support.py tests/test_m3w_opendd_pair_disagreement.py tests/test_m3w_vehicle_pair_learning.py
```

`fresh_run`: exhaustive diagnostics on all 126 paired directions.
`cached_verified`: forecasts, cohorts and original pair decisions.
Fifteen focused tests passed, including direct brute-force agreement
under complete, partial and wholly missing labels. Per-direction
checkpoints are local. No new training, threshold, deployment,
manuscript edit, external outcome access, Stage5C or SMC.

[Protocol](protocol.md) | [Counts and location intervals](readout.json).
