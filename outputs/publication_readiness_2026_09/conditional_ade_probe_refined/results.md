# Conditional Mean Versus Conditional ADE Decision

Same frozen forests, fit-scene folds, features, training labels and fixed0.9 gate. No new model fit.
All18 settings retained. Metrics below are seed means, not independent scene estimates.

| Held source | Features | Mean gain | Median gain | Mean gated gain | Median gated gain | Median native still harm |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ETH | pooled | -3.305884% | +0.000000% | +0.000000% | +0.000000% | 0.000000000 |
| ETH | scene | -6.918280% | +0.000000% | +0.000000% | +0.000000% | 0.000000000 |
| ETH | scene_neighbor | -4.364037% | +0.000000% | +0.000000% | +0.000000% | 0.000000000 |
| Hotel | pooled | -346.695087% | -90.380138% | -95.048893% | -65.126562% | 0.023124387 |
| Hotel | scene | -213.065765% | -21.370852% | -1.329899% | -0.198405% | 0.006003645 |
| Hotel | scene_neighbor | -252.225065% | -21.251070% | -0.955842% | -0.364933% | 0.005849105 |

## Every Setting

| Held | Seed | Features | Median gain | Gated gain | Median intervention | Gated intervention | Zero-optimal steps | Zero optimum with <half zero mass |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 17 | pooled | +0.000000% | +0.000000% | 0.0000% | 0.0000% | 972/972 | 0 |
| 0 | 17 | scene | +0.000000% | +0.000000% | 0.0000% | 0.0000% | 972/972 | 0 |
| 0 | 17 | scene_neighbor | +0.000000% | +0.000000% | 0.0000% | 0.0000% | 972/972 | 0 |
| 0 | 29 | pooled | +0.000000% | +0.000000% | 0.0000% | 0.0000% | 972/972 | 0 |
| 0 | 29 | scene | +0.000000% | +0.000000% | 0.0000% | 0.0000% | 972/972 | 0 |
| 0 | 29 | scene_neighbor | +0.000000% | +0.000000% | 0.0000% | 0.0000% | 972/972 | 0 |
| 0 | 43 | pooled | +0.000000% | +0.000000% | 0.0000% | 0.0000% | 972/972 | 0 |
| 0 | 43 | scene | +0.000000% | +0.000000% | 0.0000% | 0.0000% | 972/972 | 0 |
| 0 | 43 | scene_neighbor | +0.000000% | +0.000000% | 0.0000% | 0.0000% | 972/972 | 0 |
| 1 | 17 | pooled | -86.749144% | -58.463454% | 57.3944% | 11.6197% | 2767/3408 | 445 |
| 1 | 17 | scene | -21.387452% | -0.285668% | 48.5915% | 0.7042% | 2990/3408 | 463 |
| 1 | 17 | scene_neighbor | -18.647213% | -0.385945% | 49.6479% | 1.7606% | 2996/3408 | 353 |
| 1 | 29 | pooled | -93.304339% | -61.168423% | 57.3944% | 11.6197% | 2756/3408 | 472 |
| 1 | 29 | scene | -22.116725% | +0.000000% | 47.5352% | 0.0000% | 2995/3408 | 443 |
| 1 | 29 | scene_neighbor | -24.783263% | -0.337314% | 49.6479% | 2.1127% | 2959/3408 | 395 |
| 1 | 43 | pooled | -91.086932% | -75.747809% | 59.5070% | 14.0845% | 2737/3408 | 405 |
| 1 | 43 | scene | -20.608378% | -0.309546% | 44.7183% | 0.7042% | 2995/3408 | 422 |
| 1 | 43 | scene_neighbor | -20.322734% | -0.371541% | 48.9437% | 1.4085% | 3000/3408 | 356 |

Numerical refinement: 23 flagged steps, 1 still uncertified, 8 exact support-atom solutions.
The original approximate results are preserved separately. Refinement uses training labels only.
Easy percentage ratios remain undefined because still-row CV error is zero; absolute native harm is shown.
No independent-site inference, new deployment, metric/seconds claim, Stage5C or SMC.
