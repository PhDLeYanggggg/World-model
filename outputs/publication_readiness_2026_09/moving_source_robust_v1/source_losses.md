# Per-Source Fitting Costs

Matched robust-minus-mean continuation; reductions are positive when better.
Sources below are fitting sources, not independent held-out tests.

| Context | Source | Queries | Objective reduction % | Joint-cost reduction % | False-safe-cost reduction % | False-reject-cost reduction % |
|---|---|---:|---:|---:|---:|---:|
| critic0 | moving_source_component_048 | 25 | -6.5935 | -7.6614 | 0.0000 | -7.6809 |
| critic0 | moving_source_component_067 | 27 | 2.1492 | 1.7209 | -5.5850 | 1.8113 |
| critic1 | moving_source_component_053 | 39 | -8.9043 | -12.7900 | 20.8755 | -13.1094 |
| critic1 | moving_source_component_065 | 46 | -2.0595 | -29.8558 | NA | -29.8558 |
| critic1 | moving_source_component_076 | 52 | 6.9365 | 1.3834 | 34.6989 | 0.0614 |
| critic2 | moving_source_component_059 | 156 | 1.9804 | 2.6293 | -16.9983 | 3.2119 |
| critic2 | moving_source_component_060 | 630 | -4.9186 | -0.6487 | -7.4711 | -0.3314 |
| critic2 | moving_source_component_073 | 112 | -3.7611 | -1.4365 | -96.9206 | 0.3035 |

A zero control denominator produces NA, never a claimed perfect reduction.
Query counts describe the sampler units, not independent real-world scenes.
