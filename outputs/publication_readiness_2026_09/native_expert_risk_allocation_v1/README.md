# Which agents should receive a different forecast?

I kept the five causal motion experts and their equal-weight pairs
fixed. Each nonidentity action changes an OLS4 forecast. A neural
head learns the gain of that change, its positive harm, and harm on
easy windows from eight observed steps only. The other head learns
gain alone, with the same training draws and architecture. Easy-label
training cuts are computed only in each training fold. These are
fitting-source comparisons, not a new deployment calibration.

## Fixed-Count Comparison

Fresh arm64 Torch training: 54 fits, nine held sources, three seeds,
500 updates each. All active arms switch exactly 2,352 times per seed
across 1,176 queries (two per query). Equal counts do not control the
size of each change. The intervals are paired across sources, with
3,000 draws after seed averaging. Utility is measured in points on
the observed CV error denominator, not percentage ADE reduction.

| Direct contrast | Available labels | Full-grid lower gain [95% CI] |
| --- | ---: | ---: |
| Risk-adjusted independent vs OLS4 | +0.0492 | -0.0444 [-0.1215, +0.0124] |
| Risk-adjusted vs minimum change | +0.0416 | -0.0659 [-0.1405, -0.0140] |
| Risk-adjusted vs backtest | -0.0220 | -1.0353 [-1.9874, -0.3490] |
| Risk-adjusted vs same head using gain only | -0.0107 | -0.0478 [-0.0847, -0.0206] |
| Joint vs independent | +0.00007 | -0.0141 [-0.0228, -0.0064] |
| Joint vs pair-additive | -0.00054 | -0.0020 [-0.0043, -0.0005] |
| Fixed-count hindsight vs OLS4 | +3.3662 | +3.2732 [+2.0566, +4.5773] |

Available-label ADE/FDE are 91.556/234.315 pixels for independent,
91.557/234.316 for joint, and 91.597/234.452 for OLS4. The hard
contribution of the risk-adjusted arm is -0.0015 points, not an
improvement. The joint term reduces its geometric contraction proxy
against independent, but relative-vector error on observed pair steps
does not improve over OLS4. It changes 11, 11 and 19 queries versus the
pair-additive control. The optimized proxy is not a collision label.

## Where The Gap Comes From

At fixed count, hindsight selects different recipients on most
queries. Of the 3.3177-point total gap, 2.3431 [1.4866, 3.1966]
comes from recipient selection and 0.9746 [0.4526, 1.6726] from
choosing the wrong expert for the selected agents. Holding the
learned recipients but selecting their best experts yields +0.9302
[+0.4189, +1.6421] versus OLS4 in hindsight only. This is not evidence
that an available model can recover that gain.

Easy positive harm added to OLS4 peaks at 0.41% to 0.79% across
seeds on supported known-easy sources. Against the original CV
reference it still exceeds 262% on the worst supported fitting source
because OLS4 replaces CV as the default forecast. One source has no
easy denominator. The OLS4 diagnostic cannot
replace the original recording rule; no safety or deployment gate passed.

The next experiment should isolate recipient ranking before adding
more joint terms: use the same fixed menu and exact counts, test a
cross-source recipient objective and a matched deterministic control,
then test expert choice conditional on the same recipients. Do not
use this exposed set or an oracle as external calibration.

Joint metrics for multi-agent forecasting are motivated by
[Weng et al.](https://openaccess.thecvf.com/content/ICCV2023/html/Weng_Joint_Metrics_Matter_A_Better_Standard_for_Trajectory_Forecasting_ICCV_2023_paper.html).
This experiment instead studies choices among fixed forecasts;
it does not reproduce their JADE/JFDE benchmarks. The empirical
recording checks do not imply the expected-risk guarantee in
[conformal risk control](https://openreview.net/pdf?id=33XGfHLtZg).

```sh
.venv-pytorch/bin/python scripts/run_m3w_expert_risk_allocation.py train
.venv-pytorch/bin/python scripts/run_m3w_expert_risk_allocation.py score
.venv-pytorch/bin/python scripts/run_m3w_expert_risk_allocation.py replay
.venv-pytorch/bin/python scripts/report_m3w_expert_allocation_regret.py
```

Training took 34.1 seconds, with four CPU compute threads, no
loader workers and a checkpoint every 100 updates. Use
`train --resume` after interruption. Seven targeted tests passed;
the scoring and regret readouts each replayed exactly.
[Protocol](protocol.md) | [Results](analysis.json) |
[Regret attribution](regret_diagnostic.json).

No external calibration or confirmation data were opened; the
original 2% rule, the sealed 30 groups, and deployment are unchanged.
Pixels and annotation steps are not verified physical units.
No true 3D, foundation, Stage5C or SMC claim.
