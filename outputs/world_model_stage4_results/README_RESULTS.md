# Physical World Model 2.5D Stage 4 Results Package

## Folder

`/Users/yangyue/Downloads/World/outputs/world_model_stage4_results`

## What This Package Contains

### `reports/`

- `report_stage4_real_benchmark.md`  
  Main Stage 4 real long-trajectory benchmark report.

- `world_model_gate_stage4.md`  
  Model gate report. Current result: `2/7` gates passed.

- `stage4_failure_analysis.md`  
  Honest failure analysis for real t+100, learned dynamics, SMC coverage, and variable contribution.

- `metrics_stage4_real.json`, `metrics_stage4_real.csv`, `metrics_table_stage4_real.md`  
  Machine-readable and human-readable metrics.

- `real_data_episode_summary.json`  
  Real TGSIM episode construction summary.

### `source_snapshots/`

Stage 4 runnable entry points and loaders:

- `run_stage4_real_benchmark.py`
- `run_tgsim_preview.py`
- `real_trajectory_loader.py`
- `tgsim_loader.py`
- `trajnet_loader.py`
- `eth_ucy_loader.py`
- `sdd_loader.py`
- `build_real_episodes.py`
- `train_real_benchmark.py`
- `evaluate_real_benchmark.py`
- `world_model_gates.py`

### `data_summary/`

- `real_data_episode_summary.json`

## What Was Done

1. Added real trajectory loaders for TGSIM, TrajNet++, ETH/UCY, and SDD-style data.
2. Connected a public TGSIM Foggy Bottom CSV endpoint through the TGSIM loader.
3. Built verified real t+100 episodes from TGSIM trajectories.
4. Trained a real-data linear learned residual dynamics model.
5. Benchmarked constant velocity, hand physics, learned residual, stochastic residual, and SMC variants.
6. Added Stage 4 world-model gates for real data, verified horizon, learned dynamics, coverage, physical validity, semantic diversity, and audit score.
7. Fixed the real-data coordinate-origin bug so TGSIM metric coordinates are evaluated in local scene coordinates instead of being clamped incorrectly.

## Current Result

The project now has a verified real long-trajectory benchmark:

```text
dataset = TGSIM Foggy Bottom
tracks = 119
frames = 24941
samples_t100 = 482
t100_verified = true
```

But the model did not pass the Stage 4 gate:

```text
passed_gates = 2 / 7
expert_audit_score = 58 / 100
verdict = prototype_with_major_failures
```

Key metrics:

```text
constant_velocity FDE@100 = 1.00923 m
hand_physics FDE@100 = 20.05663 m
best_learned_residual FDE@100 = 33.00006 m
physics_plus_neural_residual_SMC minFDE@16@100 = 31.96071 m
coverage_FDE_lt_5m = 0.0
```

## Honest Conclusion

This is still a `pseudo-3D physics-informed learned residual state-space world model`.

It is not true 3D.

It is not yet an exceptional world model.

Stage 4 proves something useful: the real t+100 benchmark is now connected, but it exposes that the current learned residual and SMC proposal are not good enough on real long trajectories.

## Next Commands

Preview TGSIM:

```bash
python run_tgsim_preview.py /path/to/TGSIM-Foggy-Bottom-Data.csv 50000
```

Run Stage 4 benchmark:

```bash
python run_stage4_real_benchmark.py --dataset tgsim --data /path/to/TGSIM-Foggy-Bottom-Data.csv --quick
```

Public quick endpoint used in this run:

```bash
python run_stage4_real_benchmark.py --dataset tgsim --data 'https://data.transportation.gov/resource/brzy-6zfh.csv?$limit=50000' --quick
```

## Next Best Step

Do not enter a larger latent generative Stage 5 yet. First fix real-data learned dynamics with multi-step rollout loss, type-specific models, real scene geometry/goal labels, and an intent-aware SMC proposal.
