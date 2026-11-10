# Equal replacement count removes most of the apparent advantage

The [interval-supervision experiment](../opendd_vehicle_interval_angle_v1/README.md)
found 20 qualifying replacements instead of 11, but proposed 145
instead of 89. I matched the replacement count separately in each
location/opposite-location/seed direction, taking the smaller count
and keeping each arm's highest frozen scores. The original rotation
mask, targets, angle menu and risk criteria are unchanged.

| Original rotation mask | Replacements per arm | Direct qualified | Interval qualified | Interval random expectation | Hindsight ceiling |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 526 | 78 | 11 | 13 | 13.98 | 38 |
| 1,368 | 148 | 16 | 20 | 18.65 | 78 |

The random control draws from each model's own positive proposals
without changing its proposed angle; its expected count is exact.
For the direct arm the corresponding expectations are 10.50 and
15.25. Hindsight can choose either angle in the fixed menu, but has
the same count in every direction and is not a learned policy.
At the primary budget, the arms agree on 54 of the selected recordings
and agree on both the recording and angle in 49. Of 126 directions,
89 have zero matched replacements and stay in the location summary.

| Interval minus direct at equal count | 526 mask | 1,368 mask |
| --- | ---: | ---: |
| Conservative trajectory effect (pp) | +0.00000167 | +0.00001365 |
| 95% paired location bootstrap (pp) | [-0.00000130, +0.00000462] | [-0.00000777, +0.00004661] |
| Full-grid relative-motion bounds (native units) | [-0.00001254, +0.00002963] | [-0.00012036, +0.00014401] |
| Strict passes, interval vs. direct | 53 vs. 54 | 104 vs. 104 |

At the primary budget the relative-motion contribution from observed
pair-steps is positive (+0.00000855 native units; location interval
[+0.00000251, +0.00001516]). The missing-label interval still crosses
zero. The equal-count trajectory interval also crosses zero. Most of
the extra qualifying replacements disappear when intervention count
is controlled, and the interval rule does not reliably rank above
its own random subset. At this budget the finite menu still has room
for a better ranker: 38 replacements could qualify in hindsight.

The full-direction score sequence is available for this retrospective
comparison, not to an online policy. It changes no calibration or
deployment decision. The 126 directions reuse seven locations, so
the [readout](readout.json) uses location-level resampling (3,000 draws)
and does not count directions or views as independent sources. No model
was refit and no new external outcomes were opened.

[Protocol](protocol.md) |
[v191 draft](../evidence_manuscript_v191_equal_replacements/README.md)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_interval_matched_replacements.py
```
