# Stage43-BX Latent Risk Head Robustness Audit

- source: `fresh_stage43_bx_latent_risk_head_robustness_audit`
- result_source: `fresh_checkpoint_replay_latent_risk_head_robustness`
- verdict: `stage43_bx_latent_risk_head_robustness_pass_horizon_caveat`
- gate: `12 / 12`
- deployable policy changed: `False`
- protected multimodal latent-state candidate: `True`

## Global Risk Heads

| head | rows | positive rate | AUROC | AUPRC | ECE | bootstrap AUROC 95% CI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `failure` | `89736` | `37.66%` | `0.8709` | `0.8103` | `0.0387` | `[0.8626, 0.8786]` |
| `gain` | `89736` | `60.09%` | `0.8845` | `0.9298` | `0.0337` | `[0.8769, 0.8912]` |
| `harm` | `89736` | `53.28%` | `0.9050` | `0.8841` | `0.0383` | `[0.8981, 0.9116]` |

## Weak Horizon Caveats

- weak horizon slice count: `5`

| head | horizon | rows | positive rate | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| `failure` | `100` | `18070` | `22.12%` | `0.6565` | `0.2901` |
| `gain` | `100` | `18070` | `22.56%` | `0.6916` | `0.4147` |
| `gain` | `50` | `21754` | `44.35%` | `0.6923` | `0.6259` |
| `harm` | `100` | `18070` | `87.92%` | `0.6147` | `0.9016` |
| `harm` | `50` | `21754` | `68.08%` | `0.7446` | `0.8479` |

## Per-Domain Robustness

### failure

| slice | rows | positive rate | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `70585` | `37.16%` | `0.8722` | `0.8154` | `0.0376` |
| `TrajNet` | `9611` | `33.96%` | `0.8267` | `0.7022` | `0.0869` |
| `UCY` | `9540` | `45.09%` | `0.9151` | `0.9133` | `0.0866` |

### gain

| slice | rows | positive rate | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `70585` | `61.02%` | `0.8779` | `0.9300` | `0.0375` |
| `TrajNet` | `9611` | `53.26%` | `0.8464` | `0.8804` | `0.0711` |
| `UCY` | `9540` | `60.08%` | `0.9698` | `0.9817` | `0.0941` |

### harm

| slice | rows | positive rate | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `70585` | `52.66%` | `0.8933` | `0.8635` | `0.0355` |
| `TrajNet` | `9611` | `63.04%` | `0.9387` | `0.9578` | `0.1105` |
| `UCY` | `9540` | `48.03%` | `0.9701` | `0.9611` | `0.0861` |

## Per-Horizon Robustness

### failure

| slice | rows | positive rate | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `10` | `26132` | `60.77%` | `0.9098` | `0.9194` | `0.1023` |
| `100` | `18070` | `22.12%` | `0.6565` | `0.2901` | `0.1038` |
| `25` | `23780` | `45.08%` | `0.8433` | `0.7454` | `0.0532` |
| `50` | `21754` | `14.70%` | `0.7561` | `0.3351` | `0.0705` |

### gain

| slice | rows | positive rate | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `10` | `26132` | `85.43%` | `0.9370` | `0.9873` | `0.0719` |
| `100` | `18070` | `22.56%` | `0.6916` | `0.4147` | `0.0493` |
| `25` | `23780` | `75.16%` | `0.9211` | `0.9733` | `0.0866` |
| `50` | `21754` | `44.35%` | `0.6923` | `0.6259` | `0.0701` |

### harm

| slice | rows | positive rate | AUROC | AUPRC | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `10` | `26132` | `29.55%` | `0.9432` | `0.9047` | `0.0782` |
| `100` | `18070` | `87.92%` | `0.6147` | `0.9016` | `0.0520` |
| `25` | `23780` | `39.49%` | `0.9319` | `0.9026` | `0.0736` |
| `50` | `21754` | `68.08%` | `0.7446` | `0.8479` | `0.0522` |

## Latent State

- latent dim: `32`
- min variance: `0.079673`
- mean variance: `0.422024`

## Interpretation

- Stage43-BX fresh-replays the Stage43-M latent checkpoint and audits failure/gain/harm heads across domain and horizon slices.
- Global and per-domain risk heads are strong; horizon 50/100 remains weaker and is explicitly reported as a caveat.
- This strengthens the protected latent world-state evidence, but it is not an ungated policy and does not execute Stage5C or SMC.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, or foundation claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_m_checkpoint_replayed` | `True` |
| `stage43_y_precondition_seen` | `True` |
| `fresh_test_predictions_completed` | `True` |
| `latent_noncollapse` | `True` |
| `global_failure_gain_harm_heads_strong` | `True` |
| `per_domain_heads_robust` | `True` |
| `per_horizon_heads_supported` | `True` |
| `bootstrap_ci_completed` | `True` |
| `weak_horizon_caveats_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
