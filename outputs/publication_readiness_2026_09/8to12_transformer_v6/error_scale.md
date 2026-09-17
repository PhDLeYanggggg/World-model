# Fixed-Result Error-Scale Decomposition

All registered seeds, unchanged predictions/labels/metric. Equal physical-scene weighting; raw errors stay recording-local.

| Seed | Primary net excess | Floor-scale queries % | Positive harm attributable to floor-scale rows % | Floor net-excess contribution | Other rows contribution |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.34849 | 10.881 | 89.34679512322863 | 0.329733 | 0.0187576 |
| 29 | 0.467021 | 10.881 | 91.58386852956177 | 0.448992 | 0.0180289 |
| 43 | 0.407061 | 10.881 | 88.87831172142647 | 0.378072 | 0.028989 |

The two signed net-excess contributions sum to the original primary error difference. The 0.001 cutoff is the already-used numerical normalization floor, not a new physical threshold or a sample-exclusion rule.
This locates error mass, not the causal reason a network fails. It cannot prove that changing normalization improves an independently evaluated method. Better native-coordinate results are sensitivity evidence, not a replacement primary result.
Future labels are used only for this post-run diagnosis. No model, support rule, loss, threshold or deployment choice is modified. No scene-level confidence interval is inferred from one physical development site.
