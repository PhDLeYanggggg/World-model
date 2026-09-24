# Paired Joint-Decision Contrasts

Fresh arithmetic on hash-verified frozen predictions and decisions; no refit, threshold change or reserved readout.
Every claimed matched query was rechecked from its actual binary decisions, not just a solver-success label.

| Seed / cost head | Comparison | Equal-locality ADE gain (%) | Conditional 95% CI |
|---|---|---:|---|
| 17_ridge | joint_vs_independent_full_population | 0.000956 | [-0.004380, 0.007473] |
| 17_ridge | joint_exact_vs_independent_matched_nonzero | 0.001776 | [-0.004418, 0.010096] |
| 17_ridge | joint_exact_vs_unary_matched_nonzero | 0.000152 | [0.000000, 0.000455] |
| 17_neural_underharm4 | joint_vs_independent_full_population | 0.000278 | [-0.000238, 0.001070] |
| 17_neural_underharm4 | joint_exact_vs_independent_matched_nonzero | 0.000000 | [0.000000, 0.000000] |
| 17_neural_underharm4 | joint_exact_vs_unary_matched_nonzero | -0.000192 | [-0.000575, 0.000000] |
| 29_ridge | joint_vs_independent_full_population | -0.042229 | [-0.149636, 0.017878] |
| 29_ridge | joint_exact_vs_independent_matched_nonzero | -0.108901 | [-0.334242, 0.011333] |
| 29_ridge | joint_exact_vs_unary_matched_nonzero | 0.002055 | [0.000000, 0.006166] |
| 29_neural_underharm4 | joint_vs_independent_full_population | 0.000000 | [0.000000, 0.000000] |
| 29_neural_underharm4 | joint_exact_vs_independent_matched_nonzero | 0.000000 | [0.000000, 0.000000] |
| 29_neural_underharm4 | joint_exact_vs_unary_matched_nonzero | 0.000000 | [0.000000, 0.000000] |
| 43_ridge | joint_vs_independent_full_population | -0.052868 | [-0.154451, 0.002040] |
| 43_ridge | joint_exact_vs_independent_matched_nonzero | -0.098659 | [-0.277759, -0.000215] |
| 43_ridge | joint_exact_vs_unary_matched_nonzero | 0.003286 | [-0.001293, 0.011151] |
| 43_neural_underharm4 | joint_vs_independent_full_population | 0.000062 | [0.000000, 0.000187] |
| 43_neural_underharm4 | joint_exact_vs_independent_matched_nonzero | 0.000000 | [0.000000, 0.000000] |
| 43_neural_underharm4 | joint_exact_vs_unary_matched_nonzero | 0.000000 | [0.000000, 0.000000] |

Nonzero matching is fixed by predictions before target costs. Undefined ratios indicate missing supported localities or zero reference, not a zero improvement.
The unrestricted joint/independent comparison can differ in coverage; only the exact-count contrasts isolate geometry at the same intervention count.
All intervals are source-development diagnostics with shared predictors, not independent confirmation or a physical-safety claim.
