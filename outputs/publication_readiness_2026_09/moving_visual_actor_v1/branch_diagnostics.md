# Posthoc Image-Branch Diagnosis

All fourteen donors remain. These diagnostics inspect past inputs and
frozen checkpoints after the primary screen, not new targets. Clipping
or saturation associations are not causal explanations or retrained
ablations. Fitting loss is not validation performance.

| Source | Motion - quality known pp | Quality clipped inputs % | Motion clipped inputs % | Quality saturation % | Motion saturation % |
|---|---:|---:|---:|---:|---:|
| moving_source_component_047 | -0.6381 | 0.0870 | 0.0010 | 3.4861 | 0.9327 |
| moving_source_component_048 | -2.9216 | 0.0000 | 0.0000 | 0.0491 | 0.0000 |
| moving_source_component_051 | -5.7825 | 4.8436 | 8.2588 | 0.0000 | 16.2139 |
| moving_source_component_052 | -2.3767 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| moving_source_component_053 | +5.1228 | 0.4130 | 0.4474 | 0.0770 | 0.5057 |
| moving_source_component_062 | -0.0390 | 0.7484 | 1.6486 | 0.1969 | 4.2695 |
| moving_source_component_065 | +2.7523 | 0.0042 | 0.0000 | 0.0000 | 0.0000 |
| moving_source_component_067 | -26.9379 | 0.0000 | 0.0000 | 57.2103 | 3.8591 |
| moving_source_component_076 | +1.4005 | 0.0338 | 0.1815 | 0.0000 | 1.2717 |
| moving_source_component_077 | +0.1664 | 0.0007 | 0.0000 | 0.0245 | 0.0000 |
| moving_source_component_078 | -9.5115 | 0.0188 | 0.1292 | 8.6384 | 2.0771 |
| moving_source_component_080 | +4.2701 | 1.7393 | 0.9615 | 6.3695 | 0.2484 |
| moving_source_component_082 | +4.8838 | 0.1558 | 0.0569 | 0.1667 | 0.0120 |
| moving_source_component_087 | +4.3575 | 1.4004 | 0.4520 | 17.0794 | 8.9261 |

| Fitting design | Quality final minibatch loss | Motion final minibatch loss |
|---|---:|---:|
| eqmotion_A0_inner0_seed17 | 0.657554 | 0.641947 |
| eqmotion_A0_inner1_seed17 | 0.434086 | 0.412164 |
| eqmotion_A0_inner2_seed17 | 0.524329 | 0.546482 |
| eqmotion_A1_inner0_seed17 | 0.612972 | 0.617225 |
| eqmotion_A1_inner1_seed17 | 0.671975 | 0.659169 |
| eqmotion_A1_inner2_seed17 | 0.676087 | 0.671389 |

All twelve trained final projections are nonzero:True. The offset is bounded
in latent feature units, not image trajectory error. A nonsaturated
branch can still be harmful; a large input clip fraction does not prove
it causes the held-source error. No new deployment or risk-rule change.
