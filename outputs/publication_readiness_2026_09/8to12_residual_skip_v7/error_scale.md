# Fixed-Result Error-Scale Decomposition

All registered seeds, unchanged predictions/labels/metric. Equal physical-scene weighting; raw errors stay recording-local.

| Seed | Primary net excess | Floor-scale queries % | Positive harm attributable to floor-scale rows % | Floor net-excess contribution | Other rows contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.0342069 | 10.881 | 91.62216772334925 | 0.0367595 | -0.00255265 |
| 29 | 0.041304 | 10.881 | 89.27282887437593 | 0.0445504 | -0.0032464 |
| 43 | 0.0335963 | 10.881 | 91.93441331898295 | 0.0356361 | -0.00203982 |

The two signed net-excess contributions sum to the original primary error difference. The 0.001 cutoff is the already-used numerical normalization floor, not a new physical threshold or a sample-exclusion rule.
This locates error mass, not the causal reason a network fails. It cannot prove that changing normalization improves an independently evaluated method. Better native-coordinate results are sensitivity evidence, not a replacement primary result.
Future labels are used only for this post-run diagnosis. No model, support rule, loss, threshold or deployment choice is modified. No scene-level confidence interval is inferred from one physical development site.
