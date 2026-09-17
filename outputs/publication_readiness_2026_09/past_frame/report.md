# Past-Only Coordinate Frame Comparison

36 new fits and 18 exact-replayed original controls. Same model/loss/budget and all11,966 fit rows.
No final-test or policy selection. Quarter-turn gaps measure coordinate consistency, not accuracy.

| Features / arm | Gain vs CV (%) | Gain vs guard (%) | Safe positive fits | Maximum rotation gap |
| --- | ---: | ---: | ---: | ---: |
| quality_control_original | -0.99582 | -0.01269 | 0/9 | 2.49 |
| quality_control_guard_only | -0.98300 | 0.00000 | 0/9 | 2.49 |
| quality_control_past_frame | -0.86256 | 0.11928 | 0/9 | 0 |
| directed_original | -0.97244 | 0.00057 | 0/9 | 2.49 |
| directed_guard_only | -0.97302 | 0.00000 | 0/9 | 2.47 |
| directed_past_frame | -0.88781 | 0.08440 | 0/9 | 0 |

Inputs are re-expressed vector summaries, not a raw-image rotation audit. No new physical-time/metric/deployment claim.
