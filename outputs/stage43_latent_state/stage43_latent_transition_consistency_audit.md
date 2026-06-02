# Stage43-BY Latent Transition Consistency Audit

- source: `fresh_stage43_by_latent_transition_consistency_audit`
- result_source: `fresh_checkpoint_replay_latent_transition_consistency`
- verdict: `stage43_by_latent_transition_consistency_pass_with_readout_caveat`
- gate: `13 / 13`
- deployable policy changed: `False`
- protected multimodal latent-state candidate: `True`

## Global Transition Metrics

| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | `89736` | `0.7450` | `-0.0357` | `0.7456` | `0.0148` | `0.5058` |

Raw transition bootstrap 95% CI:
- transition gain vs identity: `[0.7417, 0.7483]`
- transition gain vs train centroid: `[-0.0464, -0.0261]`
- cosine next-target: `[0.7423, 0.7490]`

## Train-Only Calibrated Readout

| slice | rows | gain vs calibrated identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `all` | `89736` | `-0.0177` | `0.3097` | `0.8133` | `0.8163` | `0.3371` |

Calibrated readout bootstrap 95% CI:
- transition gain vs calibrated identity: `[-0.0241, -0.0114]`
- transition gain vs train centroid: `[0.2994, 0.3200]`

## Domain Breakdown

| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `70585` | `0.7374` | `-0.0742` | `0.7444` | `0.0387` | `0.5083` |
| `TrajNet` | `9611` | `0.7386` | `-0.0894` | `0.7056` | `-0.1108` | `0.5852` |
| `UCY` | `9540` | `0.8042` | `0.2613` | `0.7949` | `-0.0350` | `0.4077` |

## Horizon Breakdown

| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | `26132` | `0.5914` | `-0.1839` | `0.6230` | `0.0906` | `0.7484` |
| `100` | `18070` | `0.7650` | `0.1011` | `0.7597` | `-0.0124` | `0.4790` |
| `25` | `23780` | `0.8273` | `0.0701` | `0.8251` | `0.0003` | `0.3476` |
| `50` | `21754` | `0.8040` | `-0.0139` | `0.7945` | `-0.0377` | `0.4096` |

## Subset Breakdown

| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `hard_failure` | `70119` | `0.7453` | `-0.0142` | `0.7468` | `0.0183` | `0.5033` |
| `easy` | `26927` | `0.8026` | `0.0086` | `0.7937` | `-0.0333` | `0.4114` |
| `non_easy` | `62809` | `0.7186` | `-0.0508` | `0.7250` | `0.0355` | `0.5463` |

## Calibrated Readout Breakdown

### Domain

| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ETH_UCY` | `70585` | `-0.0230` | `0.2580` | `0.8055` | `0.8097` | `0.3511` |
| `TrajNet` | `9611` | `-0.0088` | `0.2729` | `0.7756` | `0.7779` | `0.3906` |
| `UCY` | `9540` | `0.0358` | `0.6736` | `0.9085` | `0.9041` | `0.1802` |

### Horizon

| slice | rows | gain vs identity | gain vs train centroid | cosine next-target | cosine identity-target | MSE next-target |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `10` | `26132` | `-0.0255` | `0.1447` | `0.6828` | `0.6907` | `0.5407` |
| `100` | `18070` | `0.0205` | `0.4828` | `0.8548` | `0.8500` | `0.2756` |
| `25` | `23780` | `-0.0170` | `0.4787` | `0.9008` | `0.9024` | `0.1948` |
| `50` | `21754` | `-0.0320` | `0.2593` | `0.8399` | `0.8450` | `0.2993` |

## Weak Transition Slices

- weak transition slice count: `4`

| axis | slice | rows | gain vs identity | gain vs train centroid | cosine next-target |
| --- | --- | ---: | ---: | ---: | ---: |
| `domain` | `ETH_UCY` | `70585` | `0.7374` | `-0.0742` | `0.7444` |
| `domain` | `TrajNet` | `9611` | `0.7386` | `-0.0894` | `0.7056` |
| `horizon` | `10` | `26132` | `0.5914` | `-0.1839` | `0.6230` |
| `horizon` | `50` | `21754` | `0.8040` | `-0.0139` | `0.7945` |

## Calibrated Readout Weak Slices

- weak calibrated readout slice count: `5`

| axis | slice | rows | gain vs identity | gain vs train centroid | cosine next-target |
| --- | --- | ---: | ---: | ---: | ---: |
| `domain` | `ETH_UCY` | `70585` | `-0.0230` | `0.2580` | `0.8055` |
| `domain` | `TrajNet` | `9611` | `-0.0088` | `0.2729` | `0.7756` |
| `horizon` | `10` | `26132` | `-0.0255` | `0.1447` | `0.6828` |
| `horizon` | `25` | `23780` | `-0.0170` | `0.4787` | `0.9008` |
| `horizon` | `50` | `21754` | `-0.0320` | `0.2593` | `0.8399` |

## Latent State

- latent dim: `32`
- z_next min variance: `0.079673`
- target min variance: `0.182923`

## Interpretation

- Stage43-BY fresh-replays the Stage43-M checkpoint and audits the latent transition itself: `z_t -> z_next` against a future target latent.
- Future waypoint/full-waypoint information is used only to encode the evaluation target latent, never as inference input.
- Raw `z_next` strongly improves over raw identity `z_t`, showing the dynamics layer moves the latent toward the future target latent.
- Raw `z_next` does not beat the train target-centroid MSE baseline globally; this is reported as a caveat rather than hidden.
- A train-only calibrated readout of `z_next` beats the train target-centroid baseline, but calibrated identity `z_t` remains slightly stronger overall. This means future-state information is readable, while independent dynamics-layer advantage is still partial.
- This is latent-dynamics evidence with caveats, not an ungated deployment policy; protected safety floors remain required.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.

## Gate

| gate | passed |
| --- | --- |
| `stage43_m_checkpoint_replayed` | `True` |
| `stage43_bx_precondition_seen` | `True` |
| `fresh_transition_predictions_completed` | `True` |
| `future_target_latent_label_eval_only` | `True` |
| `latent_noncollapse` | `True` |
| `raw_transition_lift_vs_identity` | `True` |
| `calibrated_readout_lift_vs_train_centroid` | `True` |
| `bootstrap_transition_lift_supported` | `True` |
| `domain_and_horizon_breakdowns_reported` | `True` |
| `raw_centroid_and_identity_readout_caveats_reported` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
