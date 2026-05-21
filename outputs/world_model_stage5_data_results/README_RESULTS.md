# Stage 5-Data Results Package

## Folder

`/Users/yangyue/Downloads/World/outputs/world_model_stage5_data_results`

## Current Status

The model is still a pseudo-3D / 2.5D physics-informed learned residual state-space world model. It is not true 3D and not an exceptional world model.

Stage 4.5 status carried forward:

```text
expert_audit_score = 64 / 100
verdict = prototype_with_repaired_baselines_but_failed_learned_dynamics_gate
passed_gates = 4 / 8
strongest_causal_baseline = constant_turn_rate_velocity
strongest_causal_baseline_FDE@100 = 0.04820 m
best_learned_residual_FDE@100 = 1.14498 m
```

## What Stage 5-Data Did

1. Registered 26 candidate datasets across pedestrian, crowd, drone, traffic, driving, robot/multi-agent, and synthetic domains.
2. Wrote machine-readable registry files:
   - `data_registry/dataset_registry_stage5.json`
   - `data_registry/dataset_registry_stage5.csv`
   - `data_registry/dataset_registry_stage5.md`
3. Added license audit, data cards, source-health report, and dry-run download plan.
4. Added Stage 5 world-state schema and conversion scaffolding.
5. Built a quick TGSIM Stage 5 world-state sample and 32 t+100 episodes using causal_fd velocity.
6. Added baseline benchmark scaffold and carried forward the repaired TGSIM strongest causal baseline.
7. Added deterministic-first `Stage5FoundationWorldModel` scaffold, but did not claim successful foundation training.
8. Added Stage 5 gates. Current result: `3/11`, latent Stage 5 ready: `False`.

## Important Honest Result

This is a partial data lake, not yet a large-scale foundation world model.

```text
candidate_sources = 26
actual_converted_and_benchmarked_real_sources = 1
actual_verified_t100_real_sources = 1
stage5_gates = 3 / 11
latent_stage5_ready = false
expert_audit_score = 66 / 100
```

## Why Stage 5-Latent Is Still Blocked

The deterministic learned model still has not beaten the strongest causal baseline on real data. Also, the data lake does not yet contain 3 converted real datasets or 2 actually verified real t+100 datasets.

## Main Files

- `reports/report_stage5_final.md`
- `reports/world_model_gate_stage5.md`
- `reports/report_stage5_data_discovery.md`
- `reports/report_stage5_data_lake.md`
- `reports/report_stage5_baselines.md`
- `reports/model_card_stage5.md`
- `reports/data_card_stage5.md`
- `reports/failure_analysis_stage5.md`

## Commands

```bash
python run_stage5_data_discovery.py --dry-run
python scripts/download_stage5_datasets.py --dry-run
python scripts/download_stage5_datasets.py --priority-only --max-gb 20
python run_stage5_build_episodes.py --datasets tgsim --tgsim-data 'https://data.transportation.gov/resource/brzy-6zfh.csv?$limit=50000' --quick
python run_stage5_baseline_benchmark.py
python run_stage5_train_foundation.py --config configs/stage5_foundation_quick.yaml
python run_stage5_evaluate.py --checkpoint outputs/checkpoints/stage5_best.pt
```

Do not run full training by default. Do not enable latent generative or SMC until deterministic learned dynamics beats strongest causal baselines on multiple real datasets.
