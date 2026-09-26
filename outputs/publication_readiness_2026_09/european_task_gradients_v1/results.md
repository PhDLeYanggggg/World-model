# Frozen Task-Gradient Results

Fresh fitting-batch calculation; cached_verified checkpoints and source data. No new held-out readout.
288 frozen heads; 576 disposable optimizer steps, zero persisted updates. All parent checkpoint hashes unchanged.

| Features / frozen arm | Views | Negative cost cosine | Median cosine | Auxiliary virtual cost worse / better / tied |
|---|---:|---:|---:|---|
| full / cost_only | 72 | 0 | None | 3 / 0 / 69 |
| full / membership_aux | 72 | 5 | 0.18695791071629436 | 4 / 68 / 0 |
| motion_only / cost_only | 72 | 0 | None | 4 / 0 / 68 |
| motion_only / membership_aux | 72 | 2 | 0.21129136405161025 | 2 / 70 / 0 |

## Components

| Features / arm / component | Negative cosine | Median cosine | Worse / better / tied |
|---|---:|---:|---|
| full / cost_only / H | 0 | None | 2 / 1 / 69 |
| full / cost_only / H_E | 0 | None | 3 / 0 / 69 |
| full / membership_aux / H | 42 | -0.011569197487209908 | 45 / 27 / 0 |
| full / membership_aux / H_E | 29 | 0.03891504768947453 | 30 / 42 / 0 |
| motion_only / cost_only / H | 0 | None | 4 / 0 / 68 |
| motion_only / cost_only / H_E | 0 | None | 4 / 0 / 68 |
| motion_only / membership_aux / H | 45 | -0.009711394061406416 | 46 / 26 / 0 |
| motion_only / membership_aux / H_E | 30 | 0.017543445829326484 | 33 / 37 / 2 |

## Decision

```json
{
  "majority_negative_cost_cosine": false,
  "majority_virtual_cost_worse": false,
  "registered_gradient_repair_motivated": false,
  "method_improvement_proven": false,
  "deployment_changed": false,
  "independent_confirmation": false,
  "stage5c_executed": false,
  "smc_enabled": false
}
```

Dependent fitting views, not independent samples or causal explanations of the full training path.
Raw gradient conflict alone is not AdamW interference. Full assignment detail and changes in every component are retained in aggregate_metrics.json.
The cost-only control has zero membership shared gradient by construction; an undefined cosine is not evidence of alignment.
One virtual update may worsen even cost-only loss because stored momentum, curvature and the fixed batch differ from ongoing stochastic training.
No metric/seconds, true3D, foundation, physical safety, new deployment or submission-ready claim. Stage5C and SMC remain off.
