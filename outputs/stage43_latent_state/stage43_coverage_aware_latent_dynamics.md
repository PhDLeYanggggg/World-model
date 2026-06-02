# Stage43-CG Coverage-Aware Latent Dynamics

- source: `fresh_stage43_cg_coverage_aware_latent_dynamics`
- result_source: `fresh_run`
- mode: `medium`
- gate: `15 / 15`
- verdict: `stage43_cg_coverage_aware_latent_dynamics_candidate_pass`
- deploy coverage-aware latent dynamics: `True`
- checkpoint committed: `False`

## What Changed

- This run trains the Stage43 full-waypoint latent dynamics head on the repaired CE source-family coverage split.
- It uses the local coverage-aware supervision cache from Stage43-CF.
- Future endpoints and full waypoints are labels/evaluation targets only, never inference inputs.

## Claim Boundary

- Not true 3D.
- Not a foundation world model.
- Dataset-local/raw-frame 2.5D evidence only.
- No metric or seconds-level claim.
- Stage5C not executed.
- SMC not enabled.

## Protected Test Metrics vs CE Floor

- rows: `50000`
- full-waypoint ADE improvement: `51.47%`
- endpoint FDE improvement: `54.95%`
- t50 full-waypoint ADE improvement: `31.13%`
- t100 raw-frame diagnostic: `-5.51%`
- hard/failure improvement: `49.72%`
- easy degradation: `0.00%`
- switch rate: `71.25%`

## Bootstrap CI

- bootstrap n: `2000`
- all full-waypoint ADE CI: `[51.08%, 51.85%]`
- t50 full-waypoint ADE CI: `[30.47%, 31.78%]`
- hard/failure CI: `[49.26%, 50.16%]`
- easy degradation CI: `[0.00%, 0.00%]`

## Ungated Neural Diagnostic

- full-waypoint ADE improvement: `47.85%`
- t50 full-waypoint ADE improvement: `44.08%`
- hard/failure improvement: `44.89%`
- easy degradation: `0.00%`

## Interpretation

The coverage-aware latent dynamics head is deployable under its CE floor.

## Gate

| gate | passed |
| --- | --- |
| `stage43_cf_cache_ready` | `True` |
| `torch_training_fresh_run` | `True` |
| `checkpoint_not_committed` | `True` |
| `coverage_aware_train_val_test_rows_present` | `True` |
| `future_waypoints_are_labels_only` | `True` |
| `no_future_endpoint_or_central_velocity_input` | `True` |
| `no_test_goal_or_stat_leakage` | `True` |
| `latent_noncollapse` | `True` |
| `protected_eval_completed` | `True` |
| `easy_preserved` | `True` |
| `validation_policy_selected` | `True` |
| `neural_lift_or_honest_keep_floor` | `True` |
| `ungated_neural_reported` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
