# Past-Appearance Forecast Ablation

All 18 fixed-budget models completed; 3 seeds, 2 held fit scenes, 3 input arms.
No winner chosen. Positive values indicate improvement over zero/CV on this stationary subset.

| Held source | Arm | Mean unrestricted gain % | Mean guarded gain % | Mean guarded easy absolute harm | Mean Brier lift | Mean switch rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETH | geometry | -0.39145 | -0.01330 | 0.000000 | -0.356892 | 1.2346% |
| ETH | current_rgb | -7.69380 | -8.09403 | 41.502993 | -0.335131 | 65.8436% |
| ETH | past_rgb | -14.52907 | -14.97607 | 79.014463 | -0.331905 | 66.6667% |
| Hotel | geometry | -513.47859 | -132.03553 | 30.219201 | -0.044142 | 20.5399% |
| Hotel | current_rgb | -389.04297 | -167.72675 | 36.980879 | -0.072453 | 35.2113% |
| Hotel | past_rgb | -401.79574 | -172.63225 | 38.415231 | -0.069734 | 35.6808% |

Easy percentage ratios are undefined when the CV floor is zero. Absolute harm uses the unchanged parent normalization.
Native ADE/FDE, every seed, losses and missing-image support remain in metrics.json.
Fit-only, historically exposed and adaptive research; no independent scene CI or official model improvement claim.
No physical seconds/meters/pose labels, Stage5C, SMC or deployment.
