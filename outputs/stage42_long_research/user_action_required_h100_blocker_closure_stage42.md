# User Action Required: Stage42-GW H100 Blocker Closure

No h100/t100 repair is conversion-ready now. Do not count the rows below as converted, evaluated, repaired, metric, or seconds-level evidence.

## `TrajNet|100`

- domain: `TrajNet`
- action_type: `provide_legal_official_long_raw_source`
- closure_status: `hard_blocked_missing_source_support`
- hard_blocker: `missing_official_long_raw_trajnet_source`
- required_fields: `['official source URL', 'terms accepted / allowed use', 'local path', 'source identity', 'track length sufficient for raw-frame h100/t100']`
- next_action: provide a legal official long raw source before any h100/t100 repair can run
- claim_guard: Do not write h100/t100/uniform-horizon repair as complete until guarded conversion, no-leakage, source-CV, and final eval pass.

## `UCY|100`

- domain: `UCY`
- action_type: `confirm_terms_and_run_guarded_conversion`
- closure_status: `blocked_by_terms_and_conversion_readiness`
- hard_blocker: `None`
- required_fields: `['terms accepted / allowed use', 'acceptance date', 'local path confirmation', 'source identity', 'guarded conversion and no-leakage/source-CV after confirmation']`
- next_action: confirm official terms, allowed use, local path, and source identity; then run guarded conversion/no-leakage/source-CV
- claim_guard: Do not write h100/t100/uniform-horizon repair as complete until guarded conversion, no-leakage, source-CV, and final eval pass.
