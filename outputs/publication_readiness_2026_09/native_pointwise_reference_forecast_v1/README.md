# Pointwise reference harm changes the gradients, not the deployment decision

The [previous dual-reference loss](../native_dual_reference_forecast_v1/README.md) changes the gradient through wholly unobserved targets, but the two reference errors are constants on observed rows. I tested a different intervention: replacing the global positive-CV-harm penalty with positive harm against the lower observed error of CV and OLS4. The signed CV regret, missing-row debit and CV positive-easy penalty are unchanged. Both arms start at the same CV-only checkpoint for each of three seeds and continue for 300 updates on the same draws. In the nine checkpoint batches, 22 to 39 of 128 training draws had a direct additional gradient in the treatment. The comparator winner is never a forecast input.

| Comparison | Observed-label gain (pp) | Completion lower (pp) | 95% paired source CI for lower (pp) |
| --- | ---: | ---: | ---: |
| Pointwise vs CV continuation, fixed recipients | +0.063 | -0.043 | [-0.177, +0.060] |
| Pointwise vs warm start, fixed recipients | +0.067 | -0.046 | [-0.195, +0.071] |
| Pointwise vs original gain path, fixed recipients | -0.235 | -0.912 | [-1.926, -0.063] |
| Pointwise vs CV continuation, full roster | +0.793 | -0.419 | [-1.823, +0.746] |

The paired comparison retains the same 9,675 interventions across three seeds, on 34 exposed development recordings. It uses 3,000 paired source draws after averaging seeds. The small observed advantage does not survive the 0.106-point added disagreement debit in the primary contrast. Pointwise and CV continuation pass 13/102 and 14/102 recording-seed views respectively; 23 views in each arm have undefined easy risk. All-harm exceedances increase from 58 to 65, and observed easy net degradation above 2% increases from 10 to 12 views. Neither arm improves the original gain path conservatively at this recipient budget. I do not adopt the new loss for deployment.

Six Torch fits, prediction, scoring and exact aggregate replay are `fresh_run`. The warm states, development roster, comparison bank and intervention choices are `cached_verified`. Nine non-overlapping fitting sources contribute 69,746 fitting rows; forecasts freeze before scoring. This does not estimate the effect of changing who gets switched. No external outcome, metric distance, seconds, Stage5C or SMC.

```sh
runner=scripts/run_m3w_pointwise_reference_forecast.py
.venv-pytorch/bin/python "$runner" train
.venv-pytorch/bin/python "$runner" infer
.venv-pytorch/bin/python "$runner" score
.venv-pytorch/bin/python "$runner" replay
```

Use `train --resume` for an interrupted run. Seven targeted tests passed.
[Protocol](protocol.md) | [Aggregate results](analysis.json) | [Replay](replay_execution.json)
