# Reconstructed Development Tables

Cached, SHA256-verified aggregate reports; no new training or bootstrap.
Four design-exposed sites; intervals are conditional, not confirmation.

| Model / policy | ADE gain % [95% site CI] | FDE gain % | Hard gain % | Easy degradation % | Worst-site easy degradation % |
|---|---:|---:|---:|---:|---:|
| Causal CV | 0.000 [0.000, 0.000] | 0.000 | 0.000 | -0.000 | -0.000 |
| Transformer, uncontrolled | 7.633 [5.963, 9.302] | 8.645 | 10.658 | 21.710 | 32.471 |
| EqMotion K=1, uncontrolled | 11.043 [8.225, 13.398] | 12.391 | 15.401 | 35.250 | 45.903 |
| Native-cost strict | 3.067 [1.472, 5.285] | 3.341 | 4.305 | 2.500 | 6.344 |
| Fraction-cost strict | 1.658 [0.893, 2.920] | 1.786 | 0.787 | -1.589 | 0.482 |
| Intermediate-cost strict | 3.729 [2.293, 5.391] | 4.031 | 3.744 | -0.814 | 2.947 |
| Region-weighted strict | 4.098 [2.623, 5.994] | 4.411 | 4.260 | -0.575 | 2.933 |
| Region-weighted net-gain | 11.951 [9.515, 13.816] | 12.787 | 14.224 | 21.682 | 34.092 |

Positive easy degradation is harm. Negative is improvement.
These are two predictor families; comparisons across families are not isolated loss ablations.

| Fixed paired contrast | Difference (pp) | Conditional 95% CI (pp) |
|---|---:|---:|
| Region weighting minus intermediate; strict rule | 0.36871 | [0.17316, 0.60302] |
| Region weighting minus intermediate; matched count | -0.04374 | [-0.07376, -0.01372] |
| Intermediate minus native; strict rule | 0.66173 | [-0.27228, 2.18556] |
| Intermediate minus fraction; strict rule | 2.07125 | [0.66261, 4.31900] |
| Joint minus unary geometry; matched half-count | 0.00000 | [0.00000, 0.00000] |

Matched counts are diagnostic; equal realized risk is not established.

| Scene | Seed 17 easy degradation % | Seed 29 | Seed 43 |
|---|---:|---:|---:|
| coupa | -7.69884 | -6.73178 | -6.75038 |
| deathCircle | 3.28309 | 2.84343 | 2.67344 |
| gates | 3.85432 | 1.37039 | 1.22712 |
| hyang | 0.12794 | -0.36143 | -0.73609 |

The unchanged protection criterion applies to every scene/seed, not this table's mean.
The latest combined gate fails. No new model is deployed.
