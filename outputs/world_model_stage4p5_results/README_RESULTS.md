# Physical World Model 2.5D Stage 4.5 Results Package

## Folder

`/Users/yangyue/Downloads/World/outputs/world_model_stage4p5_results`

## What This Package Contains

### `reports/`

- `stage4p5_dynamics_forensics.md`  
  Unit/dt/velocity/force failure analysis.

- `velocity_audit_stage4p5.md`  
  Native vs causal vs central velocity audit.

- `agent_type_audit_stage4p5.md`  
  TGSIM agent type distribution and speed/acceleration summary.

- `metrics_stage4p5.json`, `metrics_stage4p5.csv`, `metrics_table_stage4p5.md`  
  Stage 4.5 real t+100 benchmark metrics.

- `world_model_gate_stage4p5.md`  
  Gate report. Current result: `4/8` pass, Stage 5 ready: `False`.

- `report_stage4p5_dynamics_benchmark.md`  
  Main Stage 4.5 report.

### `source_snapshots/`

Key files changed for Stage 4.5:

- `run_stage4p5_dynamics_benchmark.py`
- `inertial_residual_model.py`
- `train_inertial_residual.py`
- `baselines.py`
- `social_force.py`
- `constraints.py`
- `scene_geometry.py`
- `tgsim_loader.py`
- `real_trajectory_loader.py`
- `build_real_episodes.py`

### `tests/`

- `test_tgsim_units.py`
- `test_causal_velocity.py`
- `test_coordinate_transform.py`
- `test_rollout_integration.py`
- `test_baseline_sanity.py`

All Stage 4.5 tests passed:

```text
9 passed
```

## Current Result

Stage 4.5 repaired the dynamics audit path:

```text
official_velocity_source = causal_fd
dt_median = 0.1 s
central_fd = diagnostic only
native_velocity = diagnostic / leakage-risk audit only
```

The old Stage 4 failure was mainly caused by wrong dynamics assumptions:

1. Real TGSIM has no loaded scene geometry / exits / goal labels in this quick endpoint.
2. Human crowd social-force / goal attraction was being applied anyway.
3. Old rollout effectively treated frame step as dt=1 instead of using dataset time.
4. Learned residual was trained over a bad hand-physics baseline.

## Key Metrics

```text
constant_velocity_causal_fd FDE@100 = 0.12288 m
constant_turn_rate_velocity FDE@100 = 0.04820 m
identity_hand_physics FDE@100 = 0.12288 m
tuned_hand_physics FDE@100 = 0.12288 m
best learned residual FDE@100 = 1.14498 m
multi-step residual FDE@100 = 1442.48877 m
```

## Gates

```text
passed_gates = 4 / 8
stage5_ready = false
expert_audit_score = 64 / 100
verdict = prototype_with_repaired_baselines_but_failed_learned_dynamics_gate
```

## Honest Conclusion

The project is healthier than Stage 4 because the baselines are now causal and physically sane.

But it is still not an exceptional world model. The strongest causal baseline is still much better than the learned residual. SMC remains premature because the deterministic learned proposal is not competitive.

Do not enter latent generative Stage 5 yet.
