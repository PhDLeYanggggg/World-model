# Stage 5 Gates

Passed: `3/11`
Latent Stage 5 ready: `False`

| Gate | Pass | Evidence | Explanation | Next Fix |
| --- | --- | --- | --- | --- |
| Data Lake Gate | False | `{"actual_converted_and_benchmarked": 1, "datasets": ["TGSIM Foggy Bottom"]}` | At least 3 real datasets must be converted and benchmarked, not merely registered. | Download/convert ETH-UCY, TrajNet++, SDD or additional TGSIM. |
| Verified Horizon Gate | False | `{"actual_verified_t100_sources": 1, "datasets": ["TGSIM Foggy Bottom"]}` | At least 2 real datasets must have verified t+100 in this project. | Build real t+100 episodes for another dataset. |
| No Leakage Gate | True | `{"official_features": "causal_fd required"}` | Official benchmark must use causal features. | Keep central/native smoothed diagnostics out of official inputs. |
| Baseline Benchmark Gate | True | `{"datasets": ["TGSIM Foggy Bottom"]}` | Each converted dataset needs strongest causal baseline. | Run run_stage5_baseline_benchmark after conversion. |
| Learned Dynamics Gate | False | `{"reason": "Stage5 deterministic model not trained in Stage5-Data dry run"}` | Learned model must beat strongest causal baseline on 2 real datasets. | Train deterministic model only after data lake is bigger. |
| Cross-Dataset Generalization Gate | False | `{"reason": "not enough converted datasets"}` | Leave-one-dataset-out should not collapse. | Convert at least 3 real datasets first. |
| Physical Validity Gate | False | `{"reason": "learned model not evaluated"}` | Learned model must preserve physical validity. | Evaluate after deterministic training. |
| Multi-Step Gate | False | `{"reason": "Stage4.5 multistep residual failed"}` | Multi-step training should improve t+50/t+100. | Implement stable rollout training/curriculum. |
| Stochastic Readiness Gate | False | `{"enabled": false}` | Only enable latent/stochastic after deterministic gate passes. | Do not enable yet. |
| SMC Readiness Gate | False | `{"enabled": false}` | Only enable SMC after stochastic coverage improves. | Do not enable yet. |
| Model Card Gate | True | `{"reports": "model/data/failure cards generated"}` | Reports must exist. | Keep cards updated after real training. |
