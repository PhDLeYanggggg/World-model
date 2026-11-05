# Decision Learning Improves Fitting, Not Safe Held Utility

## Result

I retained the original evaluation rules and changed the model objective.
The six real neural fits, treatment refit, checkpoint-resume checks and exact
prediction/action/metric replays complete successfully. The scientific repair
does not pass: known held ADE gain over the physical floor is 2.6928% versus
3.6006% for MSE, hard gain 3.5649% versus 4.6257%, and worst defined easy-net
degradation 26.2254% versus 38.9046%. Neither model satisfies the full contract.

The known paired contrast is -0.9078 percentage points, with missing-label
completion bounds [-2.2534, +0.4379] pp. They span zero and are not sampling
confidence intervals. Against affine, the known contrast is -1.6305 pp and
completion bounds are [-2.9737, -0.2874] pp. One producer seed and repeatedly
exposed development sources cannot establish independent confirmation.

## What Improved

The treatment optimizes its intended fitting behavior. Using complete empirical
query weights, cost-weighted all-channel sign regret falls 91.43%, 77.19% and
45.95% versus MSE across the three contexts. Easy-channel regret falls 81.03%,
83.07% and 85.10%. False-safe easy output counts change from 76/73/952 to
23/15/43. The common signed-cost mixed objective is 86.43-88.66% below control.
Unknown-bound MSE contributions also decrease; the bound objective was not removed.
These are repeated fitting channels, not people, held efficacy or calibration.

The result is therefore not explained by a failure to run Torch or by a loss
that never changes the learned decisions. It does not prove convergence to a
global optimum. Logistic scores have a different scale from risk-regression
targets; their large cross-objective MSE is not a calibrated-risk error metric.

## What Failed

The error is redistributed across held sources rather than removed:

| Source | Known gain contrast vs MSE, pp | Treatment intervention % |
|---|---:|---:|
| 051 | +7.4951 | 57.5181 |
| 062 | +3.7634 | 51.0085 |
| 066 | -1.9381 | 0.0000 |
| 070 | -0.0861 | 3.0449 |
| 075 | +0.0197 | 0.2847 |
| 077 | -5.1728 | 2.6767 |
| 078 | -10.4354 | 9.1606 |

The first context gains utility but its worst easy-net degradation increases
from 8.2217% to 26.2254%. The second context suppresses useful actions: source
078 intervention falls from 34.1817% to 9.1606%, while known floor gain falls
from 11.0330% to 0.5976%. The third context includes an all-fallback source;
its empty same-count utility control stays NA, not safe utility evidence.
No favorable context or source is promoted.

Known all-risk violations fall from 14 to 11, but easy-risk violations rise
from 13 to 14. The treatment passes the full recorded risk/support check in
only 1/17 views. This is recording 46 in source 077: 13 selected rows out of
3,193, no unknown selected outcomes and zero observed positive harm. It is a
small genuine descriptive pass under the existing rule, not an identity
certificate or an overall model pass. Signed easy-net and positive-harm risk
are different requirements; a good net value cannot replace the risk test.

## Supported Diagnosis And Remaining Hypotheses

Supported: the new objective reduces fitting decision cost substantially but
fails safe held transfer. It is not enough to fit the decision boundary better.
The held intervention distribution differs sharply by source. The loss changes
score magnitude as well as boundaries; no magnitude-calibration claim is made.

Unresolved: causal feature extrapolation, small fitting-source coverage (2/3/3
sources), overfitting of the new boundary, and the mixture of observed sign
supervision with unknown-bound MSE. These remain hypotheses, not separately
identified causes. The prior broad normalization and priority failures remain.
This experiment does not establish that missing modalities or a larger model
would solve the problem.

The next useful diagnostic is fitting-source versus held-development feature
support and boundary stability, with all weights/actions frozen and no outcome-
selected thresholds. Any further fit should address a measured coverage or
representation defect, not repeat a weight sweep. Reserved data stay closed.
This next experiment has not been run by this report.

## Unchanged Boundaries

No deployment change, new H, identity certificate, relaxed 2% threshold or
removed support requirement. Original 24-group primary remains NA, not passed.
No independent statistical CI or new multi-seed experiment is claimed.
New neural risk training/evaluation: fresh_run. Original trajectories, routers,
warm heads and old actions: cached_verified. Their metrics were recomputed
and exactly matched. Full historical test suite: not_run this round; 26 focused
tests pass. Native obs8/pred12 image pixels, not historical SDD t+50, verified
seconds, metric, true 3D or foundation-model success. Stage5C/SMC remain off.
