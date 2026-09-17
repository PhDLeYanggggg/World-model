# Fixed-Result Error-Scale Decomposition

All registered seeds, unchanged predictions/labels/metric. Equal physical-scene weighting; raw errors stay recording-local.

| Seed | Primary net excess | Floor-scale queries % | Positive harm attributable to floor-scale rows % | Floor net-excess contribution | Other rows contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.857872 | 10.881 | 96.40239775192207 | 0.849699 | 0.00817284 |
| 29 | 0.514803 | 10.881 | 93.07767404403091 | 0.504615 | 0.0101877 |
| 43 | 0.855573 | 10.881 | 96.96907600147804 | 0.850535 | 0.00503775 |

The two signed net-excess contributions sum to the original primary error difference. The 0.001 cutoff is the already-used numerical normalization floor, not a new physical threshold or a sample-exclusion rule.
This locates error mass, not the causal reason a network fails. It cannot prove that changing normalization improves an independently evaluated method. Better native-coordinate results are sensitivity evidence, not a replacement primary result.
Future labels are used only for this post-run diagnosis. No model, support rule, loss, threshold or deployment choice is modified. No scene-level confidence interval is inferred from one physical development site.
