# Stage43 Current World-Model Gate

- source: `fresh_stage43_dj_latent_world_state_current_reconciliation`
- verdict: `stage43_dj_latent_world_state_current_reconciliation_pass`
- passed: `13 / 13`
- protected multimodal latent-state candidate: `True`
- standalone world model deployable: `False`
- t100 support-aware head deployed: `False`
- long objective complete: `False`
- Stage5C executed: `False`
- SMC enabled: `False`

## Current Public Claim

M3W currently has a protected dataset-local/raw-frame multimodal latent-state candidate. The strongest honest claim is protected world-state evidence with useful proxy heads, not a standalone ungated true-3D or foundation model.

## Key Evidence

- Protected latent-state all/t50/t100-raw/hard/easy: `17.77%` / `13.75%` / `1.82%` / `18.16%` / `0.00%`
- Proxy heads: failure AUROC `0.8648`, gain AUROC `0.8737`, harm AUROC `0.9047`, density R2 `0.8178`, interaction AUROC `0.7694`.
- Stage43-DI t100 support-aware head is safe but diagnostic; it does not replace the stronger bounded policy.

## Boundaries

- Not true 3D.
- Not a foundation world model.
- Dataset-local/raw-frame only; no metric or seconds-level claim.
- Safety floor is still required.
- Stage5C and SMC are still off.

| gate | passed |
| --- | --- |
| `protected_latent_eval_passed` | `True` |
| `protected_latent_metrics_positive_easy_safe` | `True` |
| `multimodal_head_suite_passed` | `True` |
| `latent_noncollapse` | `True` |
| `proxy_heads_strong_enough` | `True` |
| `protected_candidate_lock_passed` | `True` |
| `safety_floor_required_not_hidden` | `True` |
| `t100_support_head_passed_but_diagnostic` | `True` |
| `t100_support_head_safe_no_deployment_lift` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
