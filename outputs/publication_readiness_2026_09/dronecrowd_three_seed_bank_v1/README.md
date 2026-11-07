# Three-Seed Forecast Controls For External Evaluation

I checked whether the existing predictor library can support a new independent
8-observation/12-prediction study without retraining identical controls.

| Check | Result | Evidence source |
|---|---|---|
| Original predictor fits | 18 checkpoints, 4,000 updates each; PCT and EqMotion, three folds, seeds 17/29/43 | `cached_verified` |
| Real Torch inference replay | 18,432 A-only rows; all predictions exactly match saved arrays | `fresh_run` |
| Contributing training recordings covered by the cross-archive screen | All for folds 0/1 (12 checkpoints); fold 2 (6 checkpoints) includes 10 unscreened contributors | `fresh_run` roster audit |
| External scoring and three-seed controller | Not run; scientific roles unassigned and controller lineage not yet verified | `not_run` |

The probe keeps original contiguous minibatch boundaries and selects inputs by
past eligibility and row ID, not outcome. Each checkpoint's training exposure
is checked against the original fitted row counts. Two otherwise unscreened
short clips contribute zero eligible training rows in the covered folds. The
screen does not certify independent physical sites. Coverage of the forecasters
does not cover a later gain/harm head's training data.

The narrow-screen figures above describe that run only. A later
[combined coverage check](../dronecrowd_combined_forecaster_coverage_v1/README.md)
joins the previously completed wider screen for the ten additional training
recordings, bringing image-screen coverage to all 18 forecasters.

PID 9164 exited 0 after 37.41 s (peak RSS 868,843,520 bytes; native arm64
Torch CPU4/interop1, zero loader workers). No optimizer updates, no external
inference, no external future-label access, and no deployment change. The
original 2% risk and support rules remain unchanged; old failed views stay
failed or not estimable as previously reported. No Stage5C or SMC.

[Machine-readable results](analysis.json) |
[Execution receipt](executions/9164_1794031150026091000.json) |
[Local reproduction and next checks](operation_zh.md)
