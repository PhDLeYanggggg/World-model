# Conditional Risk Components: Fitting-Only Results

## Material Passport

Fresh component accounting and full exact replay; cached_verified packets and216 frozen heads. No parameter updates, new actions, held readout, independent calibration or deployment. All108 groups retained. Source views repeat localities; these are not216 independent samples.

The table uses the exact expected source/query-balanced marginal fitting objective over all known rows, not the128-query monitor subset used during training.

| Arm | Row risk MSE | Query-mean risk MSE | Half-sum objective |
|---|---:|---:|---:|
| uncapped | 0.00345242867 | 0.00210871486 | 0.00278057177 |
| risk_priority | 0.00352498842 | 0.00215375468 | 0.00283937155 |

Repaired heads improve full-fitting marginal objective in42/108 groups.

## Predicted-Factor Replacement Accounting

All8 combinations are scored without constructing actions. Entries average marginal loss changes over all6 replacement orders, then108 repeated groups. Positive means worse fitting. They sum to risk-priority minus uncapped loss, not to a causal explanation of held error.

| Quantity | Occurrence replacement | Reference replacement | Harm replacement |
|---|---:|---:|---:|
| row_MSE | 0.000331867166 | -8.57052635e-06 | -0.000250736891 |
| query_mean_MSE | 0.000291194219 | -6.1213726e-06 | -0.00024003303 |
| marginal | 0.000311530693 | -7.34594947e-06 | -0.00024538496 |

## Label Partitions

Partitions share the full known-query weights and add to row MSE; they are not individually renormalized here. A non-easy row has zero easy-risk target, not necessarily zero overall harm.

| Arm | Non-easy contribution | Easy zero-harm contribution | Easy positive-harm contribution |
|---|---:|---:|---:|
| uncapped | 0.000656405225 | 4.05665813e-05 | 0.00275545686 |
| risk_priority | 0.000535264917 | 2.0798945e-05 | 0.00296892456 |

## Exact Residual Identity

Order: occurrence/reference/harm. Cross order: occurrence-reference, occurrence-harm, reference-harm. Squared terms plus doubled cross terms equal MSE; negative cross terms represent cancellation. Attribution is order-dependent arithmetic, not an oracle deployable model.

- uncapped: squared=[0.003289736846975194, 2.1940925867149963e-05, 0.0011174629684054887]; doubled cross=[2.003719344453631e-05, -0.0010243072888987913, 2.7558024684012175e-05].
- risk_priority: squared=[0.005000184924519159, 3.0753487213067717e-05, 0.002979421629653794]; doubled cross=[0.00010676852063014822, -0.004508096693742779, -8.404344904147232e-05].

## Boundaries

Fitting loss is not transport performance. This diagnostic does not evaluate useful-switch ranking because signed benefit is absent from its fitting packets. No selected-risk gate is repaired by these calculations. Unknown labels remain excluded, not zero. Only already-opened development fitting roles are used. Image-local detector silver, observation8/prediction12, raw stride12. No metric, seconds, human-gold, physical-safety, true3D, foundation or submission claim. Stage5C/SMC remain disabled.
