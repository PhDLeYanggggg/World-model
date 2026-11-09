# Learning how many forecasts to replace

I changed the decision space rather than tuning the prior recipient
ranking. Each query can keep CV, intervene on 25%, 50%, 75% or
100% of its eligible rows, or keep the prior analytic portfolio
count. Each count has several whole-query plans built from a shared
multi-amplitude motion bank, using independent scores, signed risk
prices, or proximity to OLS4. Both learned scorers see the same
plans and targets. One has neighbor geometry features and one masks
those features. No future coordinate or validity mask enters their
inputs. There is no learned donor path in this study, so candidate
scope does not change across roles.

[Decision-focused ranking](https://proceedings.mlr.press/v162/mandi22a.html)
motivates training on alternative decisions rather than independent
labels. [Joint Metrics Matter](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html)
motivates measuring the neighbors' relative geometry. Here the
decision is which already-computable paths replace a CV forecast
under an unchanged risk rule. The pair-distance diagnostic is not
a collision or physical-safety claim.

## Learned policies

18 arm64 PyTorch fits, 21,600 updates, three seeds and three source
rotations. Held development: 24 recordings in nine source groups,
72 recording/seed views, and 79,047 query-plan options across those
views. These counts are not independent sample sizes.

| Policy | Edits | Risk passes / 72 | Undefined easy / 72 | Worst known signed easy |
|---|---:|---:|---:|---:|
| Analytic parent portfolio | 45,042 | 4 | 14 | +10.21% |
| Independent fixed quarter | 53,637 | 2 | 13 | +45.06% |
| Joint scorer at parent counts | 45,042 | 5 | 14 | +25.51% |
| No pair features, free counts | 42,768 | 3 | 20 | +124.35% |
| Independent at joint counts | 49,734 | 5 | 25 | +114.04% |
| No pair features at joint counts | 49,734 | 4 | 25 | +126.13% |
| Joint scorer, free counts | 49,734 | 3 | 25 | +126.59% |

The joint scorer changes 23.77% of the row predictions. It increases
the parent count in 549 queries, decreases it in 1,503 and preserves
it in 1,476; it chooses zero in 2,790 of the 3,528 query/seed views.
Editing more rows overall does not imply broad support across
recordings. The worst recording's p95 known ADE is unchanged from
CV (1,262 local pixels). No calibration cell qualifies under the
original risk checks and positive lower gain against both references.

| Contrast | Observed-only pp | Completion-lower pp (95% CI) |
|---|---:|---:|
| Joint vs independent, exact query counts | -0.054 | -0.322 [-0.660, -0.120] |
| Free vs parent count, same joint scorer | +1.021 | -0.508 [-1.268, +0.126] |
| Pair features vs no pair features, exact query counts | +0.002 | -0.170 [-0.361, -0.043] |
| Joint vs CV | +1.343 | +0.207 [-0.156, +0.659] |
| Joint vs OLS4 | +0.742 | -53.627 [-95.411, -21.994] |

Paired intervals use 3,000 source resamples after averaging the three
seeds. Free count also loses 2.119 local pixels on the conservative
pair-distance comparison [-5.244, -0.394]. Adding pair inputs loses
0.251 local pixels at equal counts [-0.636, -0.034]. Neither the
observed-only count gain nor these pair inputs supports deployment.

## Does the menu contain valid choices?

I allowed outcome-aware choices only for this diagnostic. For each
recording the linear relaxation permits mixtures of whole-query
options, keeps the three original risk inequalities and a positive
easy denominator, and maximizes the smaller of the two conservative
gains. A dual mixture bounds the integer objective from above.
I count a rounded plan only after rescoring the actual trajectories
with the original scorer. The numerical positive-gain tolerance
here is 0.0000001 pp, not a change to the policy's risk test.

| Menu | Verified witnesses | Nonpositive upper | Infeasible relaxation | No easy support |
|---|---:|---:|---:|---:|
| Parent counts, alternate compositions | 0 | 9 | 50 | 13 |
| Variable counts and alternate compositions | 8 | 52 | 6 | 6 |

Ignoring risk, at least one comparator's independent upper bound is
nonpositive in 53 fixed-count and five variable-count views. Under
the original rule, variable counts are helpful but still leave
only eight verified views. The exact-count controls also show that
the learned joint scorer chooses worse actions than independent
selection at the same counts. Both **proposal coverage under risk**
and **decision quality** remain problems. Next I would hold the
original rule fixed and build risk-diverse whole-query alternatives
from the full per-agent bank, check how many verified witnesses they
recover, then train on contrasting feasible and infeasible plans.

## Reproduction and status

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_joint_count.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_joint_count_capacity.py
```

The fits and capacity audit are `fresh_run`; the earlier shape
checkpoints are `cached_verified`. The new checkpoints reproduce
the frozen readout without optimizer updates. Remove `--replay`
to train or resume using the local lock, checkpoints and heartbeat.
The run finished locally in 102 seconds with four compute threads
and no DataLoader workers. No new CREATE job. Tests cover the exact
count, no-action fallback, pair label, ablation mask and LP checks.

Native eight observed / twelve forecast steps in dataset-local
pixels. No metric, seconds-level, true 3D, foundation or physical
safety claim. DroneCrowd's 15/30 roles are unchanged; independent
confirmation is `not_run`. No deployment, Stage5C, SMC or submission.

[Protocol](protocol.md) | [Readout](readout.json) |
[Capacity audit](capacity.json) | [Working supplement](../evidence_manuscript_v141/README.md).
