# Physical World Model 2.5D Results Package

## Folder

`/Users/yangyue/Downloads/World/outputs/world_model_stage3_results`

## What This Package Contains

### `reports/`

- `model_audit_stage2.md`  
  Honest audit of what is observed, derived, latent, assumed, hand-coded, and learned.

- `report_stage2.md`  
  Main Stage 2 experiment report with synthetic t+100 evaluation and AerialMPT limitations.

- `report_stage2_continuation.md`  
  Notes on the follow-up changes: event-balanced evaluation, latent-goal SMC, and semantic clustering fixes.

- `world_model_expert_self_audit.md`  
  Automatic expert self-audit. Current score: `58/100`, verdict: `prototype_with_major_failures`.

- `world_model_v3_target_spec.md`  
  The stricter target spec for a genuinely strong world model.

- `metrics_stage2.json`, `metrics_stage2.csv`, `metrics_table_stage2.md`  
  Machine-readable and human-readable evaluation metrics.

- `data_sources_stage2.md`, `data_sources_stage3.md`  
  Real-data source research and ranked dataset plan.

- `world_variable_schema_stage3.md`  
  New variable schema for Stage 3.

### `figures/stage2/`

Generated visual outputs:

- synthetic scene map
- ground-truth rollout
- prediction vs ground truth t+100
- multi-branch rollout
- terminal semantic clusters
- collision/density/obstacle heatmaps
- metrics comparison table
- loss curves
- rollout GIF

### `configs/`

- `stage2.yaml`  
  Stage 2 training/evaluation config.

- `stage3_data.yaml`  
  Stage 3 data-source and variable-expansion config.

- `requirements.txt`  
  Python requirements.

### `source_snapshots/`

Important runnable entry points and source files:

- `run_stage2_demo.py`
- `run_world_model_audit.py`
- `run_stage3_data_catalog.py`
- `run_tgsim_preview.py`
- `world_model_self_audit.py`
- `stage3_dataset_catalog.py`
- `tgsim_adapter.py`
- `world_state_features.py`

### `data_catalog/`

- `data_sources_stage3.json`  
  Structured dataset catalog for future loaders.

## What Was Done

1. Built a SyntheticPhysicalCrowd2.5D environment for verified synthetic t+100 evaluation.
2. Added hand-physics, deterministic neural residual, stochastic neural residual, and SMC rollout comparisons.
3. Added an expert self-audit system that scores the model against strict world-model gates.
4. Re-ran the Stage 2 demo after environment cleanup.
5. Improved physical consistency and terminal semantic diversity.
6. Added Stage 3 data-source research and ranked datasets for real long-trajectory training.
7. Added new Stage 3 feature variables:
   - time to collision,
   - closing speed,
   - directional density,
   - gap,
   - bottleneck score,
   - nearest exit distance,
   - velocity-goal alignment,
   - dwell flag,
   - heading-rate proxy,
   - obstacle tangent.
8. Added a TGSIM adapter skeleton for converting real trajectory CSVs into world-model variables.

## Current Result

The model is now a:

`pseudo-3D physics-informed learned residual state-space world model`

It is not true 3D.

It is not yet an exceptional world model.

Current expert audit:

```text
score = 58 / 100
verdict = prototype_with_major_failures
```

Strong points:

- Synthetic t+100 is verified.
- AerialMPT t+100 is correctly marked qualitative-only.
- Physical consistency improved.
- Terminal cluster diversity improved.
- Stage 3 variable expansion is now wired into the model feature extractor.

Still weak:

- Learned residual does not meaningfully beat hand physics.
- SMC does not yet improve coverage enough.
- Real long-trajectory t+100 benchmark is not connected yet.
- TGSIM/SDD/ETH/TrajNet data is cataloged but not fully downloaded/trained.

## Next Commands

Run expert audit:

```bash
python run_world_model_audit.py
```

Run Stage 2 demo:

```bash
python run_stage2_demo.py
```

Generate Stage 3 data catalog:

```bash
python run_stage3_data_catalog.py
```

Preview a local TGSIM CSV:

```bash
python run_tgsim_preview.py /path/to/TGSIM-Foggy-Bottom-Data.csv 50000
```

## Next Best Step

Download or point the project to TGSIM Foggy Bottom data, run `run_tgsim_preview.py`, then build real long-trajectory episodes for verified real t+100 evaluation.

## Stage 4 Update

Stage 4 results are now consolidated here:

`/Users/yangyue/Downloads/World/outputs/world_model_stage4_results`

What changed:

1. Added real trajectory loaders for TGSIM, TrajNet++, ETH/UCY, and SDD-style data.
2. Connected a public TGSIM Foggy Bottom CSV endpoint.
3. Built verified real t+100 benchmark episodes.
4. Compared constant velocity, hand physics, learned residual, stochastic residual, and SMC variants.
5. Added strict world-model gates.

Main Stage 4 outcome:

```text
real dataset = TGSIM Foggy Bottom
samples_t100 = 482
t100_verified = true
passed_gates = 2 / 7
expert_audit_score = 58 / 100
verdict = prototype_with_major_failures
```

The real benchmark exposed a major failure:

```text
constant_velocity FDE@100 = 1.00923 m
hand_physics FDE@100 = 20.05663 m
best learned residual FDE@100 = 33.00006 m
physics_plus_neural_residual_SMC minFDE@16@100 = 31.96071 m
coverage_FDE_lt_5m = 0.0
```

Conclusion:

The project now has a verified real t+100 benchmark, but the model is still not an exceptional world model. Learned residual does not beat hand physics on TGSIM, SMC does not improve coverage, and Stage 5 should wait until real-data dynamics and intent-aware proposals are fixed.
