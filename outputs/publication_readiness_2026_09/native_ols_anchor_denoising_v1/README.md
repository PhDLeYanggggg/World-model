# Denoising the OLS4 forecast anchor did not improve the full roster

Last-step constant velocity is a strong and noise-sensitive baseline ([Scholler et al., 2020](https://arxiv.org/abs/1903.07933)). This experiment keeps the fitted four-observation slope, but lets its forecast start move toward the fitted endpoint. It separates anchor estimation from velocity estimation; it does not reproduce the published ETH/UCY benchmark. Fixed anchor shifts, a raw line, anchored OLS8 and a maximum-displacement-matched slope change serve as controls.

I trained 54 matched Torch fits (two directions, nine held fitting sources, three seeds). Past-only coefficients freeze before scoring. The comparison covers the whole 69,746-window fitting roster with leave-one-source-out predictions, not a learned selection budget, and uses 3,000 paired source draws after averaging seeds. The full common-support anchor/slope comparison uses exactly the same learned fraction and maximum displacement on each row, with OLS4 elsewhere.

| Contrast | Conservative gain (pp on observed CV denominator) | 95% paired source interval |
| --- | ---: | ---: |
| Raw fitted line vs. OLS4 | -2.0724 | [-3.0544, -1.2651] |
| Fixed ramped anchor vs. OLS4 | -1.7454 | [-2.5110, -1.1226] |
| Fixed displacement-matched slope vs. OLS4 | -0.8594 | [-1.7060, -0.0386] |
| Learned anchor vs. OLS4 | -0.00013 | [-0.00020, -0.00007] |
| Learned anchor vs. CV | -10.9736 | [-24.7223, +1.0601] |

Available-label ADE/FDE are 91.5975/234.4517 image pixels for OLS4 and 91.5975/234.4518 for the learned anchor. The small trained coefficients explain the near equality: anchor median fraction 0.0147%, p95 0.1300%; only 748 of 209,238 row-seed views use at least 1%. The learned direction does not give a better candidate. Fixed ramped anchor is worse in all nine sources. One source has zero eligible easy windows, so the macro easy-ADE result is undefined, not dropped or interpreted as a pass. The [reporting amendment](reporting_amendment.md) preserved all predictions and definitions without re-fitting.

**Post-hoc capacity check.** Both fixed directions have negative conservative right slopes at zero (-1.6204 and -0.8366 points per amplitude); no common positive amplitude improves this fixed roster. Label-aware rowwise amplitudes can still extract +1.0436 points from the anchor direction [0.6022, 1.5296] and +0.5877 from the slope [0.2597, 0.9873]. That is small hindsight headroom, not evidence that the learned coefficient can find it. It argues against spending more effort on whole-roster anchor denoising.

Eleven related tests passed. The support-aware score and the capacity diagnostic each replay exactly. No development targets, external outcomes, official rule change, deployment, Stage5C or SMC. Units are image pixels and 8/12 annotation steps, not verified physical distance or seconds.

```sh
.venv-pytorch/bin/python scripts/run_m3w_ols_anchor_denoising.py train
.venv-pytorch/bin/python scripts/report_m3w_ols_anchor_denoising.py score
.venv-pytorch/bin/python scripts/report_m3w_ols_anchor_denoising.py replay
.venv-pytorch/bin/python scripts/report_m3w_ols_anchor_capacity.py
```

Use `train --resume` for interrupted folds. [Protocol](protocol.md) | [Aggregate outcomes](analysis.json) | [Capacity](capacity.json) | [Replay](replay_execution.json)
