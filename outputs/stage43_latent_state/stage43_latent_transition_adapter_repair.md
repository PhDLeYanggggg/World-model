# Stage43-BZ Latent Transition Adapter Repair

- source: `fresh_stage43_bz_latent_transition_adapter_repair`
- result_source: `fresh_train_only_latent_transition_adapter_repair`
- verdict: `stage43_bz_latent_transition_adapter_repair_pass`
- gate: `15 / 15`
- deployable policy changed: `False`

## Global Comparison

| model | rows | gain vs identity | gain vs train centroid | MSE next-target |
| --- | ---: | ---: | ---: | ---: |
| `Stage43-M z_next` | `89736` | `0.7450` | `-0.0357` | `0.5058` |
| `Stage43-BZ adapter z_next` | `89736` | `0.8404` | `0.3516` | `0.3167` |

## Train-Only Calibrated Readout

| model | gain vs calibrated identity | gain vs train centroid | MSE next-target |
| --- | ---: | ---: | ---: |
| `Stage43-BY calibrated z_next` | `-0.0177` | `0.3097` | `0.3371` |
| `Stage43-BZ calibrated adapter` | `0.2014` | `0.4583` | `0.2646` |

Calibrated readout bootstrap 95% CI:
- gain vs identity: `[0.1884, 0.2138]`
- gain vs train centroid: `[0.4482, 0.4695]`

## Adapter Domain Breakdown

| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | MSE next-target |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `70585` | `0.8327` | `0.3156` | `0.8277` | `0.3238` |
| `TrajNet` | `9611` | `0.8020` | `0.1747` | `0.7632` | `0.4433` |
| `UCY` | `9540` | `0.9347` | `0.7536` | `0.9297` | `0.1360` |

## Adapter Horizon Breakdown

| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | MSE next-target |
| --- | ---: | ---: | ---: | ---: | ---: |
| `10` | `26132` | `0.7498` | `0.2751` | `0.7550` | `0.4583` |
| `100` | `18070` | `0.8731` | `0.5146` | `0.8633` | `0.2586` |
| `25` | `23780` | `0.8979` | `0.4498` | `0.8917` | `0.2057` |
| `50` | `21754` | `0.8487` | `0.2177` | `0.8318` | `0.3161` |

## Interpretation

- Stage43-BZ freezes the Stage43-M past encoder and future-target encoder, then trains a past-only latent transition adapter on train rows only.
- Future target latents are label/eval targets only; they are not inference inputs.
- The adapter is not a deployable policy change and does not remove the safety floor.
- This stage directly tests whether the Stage43-BY readout caveat is caused by a weak transition head rather than by absent causal signal.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_m_checkpoint_replayed` | `True` |
| `stage43_by_precondition_seen` | `True` |
| `train_only_adapter_completed` | `True` |
| `future_target_latent_label_eval_only` | `True` |
| `no_test_statistics_normalization` | `True` |
| `latent_noncollapse` | `True` |
| `raw_adapter_beats_identity` | `True` |
| `raw_adapter_beats_stage43_m_transition` | `True` |
| `raw_adapter_beats_train_centroid` | `True` |
| `calibrated_adapter_beats_identity` | `True` |
| `calibrated_bootstrap_supports_identity_lift` | `True` |
| `domain_and_horizon_breakdowns_reported` | `True` |
| `weak_slice_caveats_reported` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
