# Every Source Fold

Exposed source-held fitting diagnostic, not independent confirmation.

| Context | Family | Source | Known rows | Queries | MSE reduction % | Joint-cost reduction % | MSE reduction vs mismatched % |
|---|---|---|---:|---:|---:|---:|---:|
| critic0 | ridge | 048 | 1769 | 25 | -0.780 | -2.242 | -1.698 |
| critic0 | extra_trees | 048 | 1769 | 25 | -2.099 | -0.001 | -1.332 |
| critic0 | ridge | 067 | 3998 | 27 | -5.086 | +0.689 | -4.658 |
| critic0 | extra_trees | 067 | 3998 | 27 | -0.695 | +0.000 | -0.598 |
| critic1 | ridge | 053 | 672 | 39 | -1.905 | +16.625 | +2.580 |
| critic1 | extra_trees | 053 | 672 | 39 | +9.363 | +0.000 | +3.621 |
| critic1 | ridge | 065 | 44 | 19 | +1.698 | +0.000 | -4.460 |
| critic1 | extra_trees | 065 | 44 | 19 | +0.129 | +0.000 | +3.047 |
| critic1 | ridge | 076 | 2999 | 52 | -1.356 | -20.117 | -0.614 |
| critic1 | extra_trees | 076 | 2999 | 52 | +0.281 | +0.863 | +0.068 |
| critic2 | ridge | 059 | 8772 | 150 | +2.760 | -0.013 | +4.471 |
| critic2 | extra_trees | 059 | 8772 | 150 | +7.159 | +0.000 | +9.539 |
| critic2 | ridge | 060 | 10749 | 603 | -0.660 | -12.154 | -0.361 |
| critic2 | extra_trees | 060 | 10749 | 603 | +0.559 | +0.000 | +0.400 |
| critic2 | ridge | 073 | 19058 | 112 | -0.284 | +3.237 | -0.211 |
| critic2 | extra_trees | 073 | 19058 | 112 | +0.218 | +0.000 | -0.213 |

All sources are retained. Very small source support and dependent queries limit inference.
Known-label probes do not certify missing-label full-policy harm. No safety pass follows from this table.
