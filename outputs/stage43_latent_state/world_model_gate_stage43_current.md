# Stage43 Current World-Model Gate

- source: `fresh_stage43_bz_latent_transition_adapter_repair`
- verdict: `stage43_bz_latent_transition_adapter_repair_pass`
- passed: `15 / 15`
- protected multimodal latent state candidate: `True`
- adapter raw gain vs identity: `0.8404`
- adapter calibrated gain vs identity: `0.2014`
- adapter calibrated CI low vs identity: `0.1884`
- deployable policy changed: `False`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Boundary

- Stage43-BZ is a latent transition repair experiment, not an ungated deployment policy.
- Safety floors remain required for deployment.
- Dataset-local/raw-frame 2.5D only; no metric, seconds-level, true-3D, foundation, Stage5C, or SMC claim.

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
