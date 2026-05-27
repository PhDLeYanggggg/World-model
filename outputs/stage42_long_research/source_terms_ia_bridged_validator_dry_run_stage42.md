# Stage42-IB IA Bridged Validator Dry Run

- source: `fresh_stage42_ib_ia_bridged_validator_dry_run`
- generated_at_utc: `2026-05-27T21:25:42.948473+00:00`
- git_commit: `88c016e`
- input_hash: `109225824f5c7f1d2154c93c2b87af081eac95737fc061b0b2d6c642dcd531d9`
- gate: `16 / 16`
- verdict: `stage42_ib_ia_bridged_validator_dry_run_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-IB explicitly validates the IA bridged intake in dry-run mode; it does not activate conversion.
- The canonical CG readiness manifest is not overwritten by this dry-run.
- local path found 不等于 legal terms accepted，不等于 official source identity confirmed。
- future endpoints / waypoints 只作为 supervised/evaluation labels，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- targets_validated: `5`
- terms_accepted_targets: `0`
- conversion_ready_targets: `0`
- blocked_targets: `5`
- converted_datasets_now: `0`
- evaluated_datasets_now: `0`
- dry_run_only: `True`
- canonical_manifest_overwritten: `False`

## Validation Table

| dataset | terms accepted | conversion ready | CF blockers | confirmation blockers |
| --- | ---: | ---: | --- | --- |
| `ucy_crowd_original` | `False` | `False` | manual_terms_or_application_required, no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, source_identity_missing |
| `eth_biwi_original` | `False` | `False` | manual_terms_or_application_required, no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, source_identity_missing |
| `trajnetplusplus_official` | `False` | `False` | manual_terms_or_application_required, no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, source_identity_missing |
| `opentraj_toolkit` | `False` | `False` | no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, source_identity_missing |
| `aerialmpt_or_other_topdown` | `False` | `False` | local_path_missing, schema_not_parseable, manual_terms_or_application_required, no_independent_t50_candidate | terms_not_accepted, terms_acceptance_date_missing, allowed_use_missing, source_identity_missing |

## Interpretation

- IB confirms the IA bridged intake is structurally compatible with CG validation semantics.
- It keeps conversion-ready targets at zero because the HZ confirmation fields remain blank.
- This dry-run writes a separate manifest and does not overwrite the canonical CG manifest.
