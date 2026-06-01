# Stage43-BI Locked Candidate Paper Package Refresh

- source: `fresh_stage43_bi_locked_candidate_paper_package_refresh`
- result_source: `fresh_package_refresh_from_stage43_bh_candidate_lock`
- verdict: `stage43_bi_locked_candidate_paper_package_refresh_pass`
- gate: `14 / 14`
- candidate: `protected_multimodal_latent_state_world_model_candidate`
- paper package refreshed: `True`

## Current Claim

M3W currently has evidence for a protected multimodal latent-state world-model candidate under a safety floor, not for a standalone ungated model.

## Latest Protected Candidate Metrics

- rows: `89736`
- all improvement: `50.25%`
- t50 improvement: `51.23%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure improvement: `47.88%`
- easy degradation: `0.00%`
- switch rate: `70.45%`

## Evidence

- deployable proxy heads: `['failure_risk', 'gain_opportunity', 'harm_guard', 'causal_history_density', 'future_interaction_risk']`
- diagnostic-only heads: `['waypoint_label_availability', 'smoothness_validity_proxy']`
- stable positive t50 ablation variants: `['no_neighbor_interaction', 'no_baseline_floor', 'no_domain']`
- external domains: `['ETH_UCY', 'TrajNet', 'UCY']`
- source-level test rows: `89736`

## Allowed Claims

- protected dataset-local/raw-frame 2.5D multi-agent world-state candidate
- multimodal latent-state heads are useful as protected proxy heads
- latest protected tail-horizon candidate improves all/t50/hard while preserving easy cases
- safety floor remains part of the method

## Disallowed Claims

- true 3D world model
- large-scale foundation world model
- metric or seconds-level prediction
- ungated standalone deployment
- uniform positive external transfer across every source
- Stage5C execution
- SMC execution

## Gate

| gate | passed |
| --- | --- |
| `candidate_lock_passed` | `True` |
| `legacy_paper_refresh_available` | `True` |
| `multimodal_head_suite_available` | `True` |
| `external_matrix_available` | `True` |
| `multiseed_ablation_available` | `True` |
| `source_terms_guard_blocks_unconfirmed_support` | `True` |
| `protected_candidate_not_standalone` | `True` |
| `latest_candidate_positive_easy_safe` | `True` |
| `paper_package_outputs_declared` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_new_training_or_conversion` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
