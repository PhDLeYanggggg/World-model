# Stage43-CL T100 Source-Stable Compatibility Audit

- source: `fresh_stage43_cl_t100_source_stable_compatibility_audit`
- result_source: `fresh_reconciliation_of_stage43_t_local_t100_and_stage43_ck_global_floor`
- gate: `12 / 12`
- verdict: `stage43_cl_t100_source_stable_compatibility_pass_local_only`
- local t100 positive signal allowed: `True`
- global t100 deployment allowed: `False`

## Compatibility Result

- Stage43-T local rows: `1440`
- current external matrix rows: `89736`
- row ratio: `0.0160`
- same split scope: `False`
- Stage43-T family: `TrajNet_crowds`
- Stage43-T test sources: `OpenTraj/datasets/TrajNet/Train/crowds/crowds_zara03.txt`
- Stage43-T h100 ADE lift: `2.59%`
- Stage43-T h100 endpoint lift: `-0.55%`
- Stage43-T easy degradation: `0.00%`
- CK global t100 diagnostic: `0.00%`
- compatibility reason: `local_source_level_positive_signal_not_current_full_matrix_deployment`

## Feature Contract Audit

- feature names persisted in Stage43-T report: `False`
- future waypoints label-only: `True`
- test threshold tuning: `False`
- denied protocol fragments: `[]`
- causal admissibility status: `report_protocol_clean_but_feature_names_not_persisted`
- promotion follow-up: `Persist feature names and source-split hashes before promoting Stage43-T beyond local source-level evidence.`

## Claim Decision

- local t100 positive signal may be reported: `True`
- global t100 deployment may be reported: `False`
- uniform t100 success may be reported: `False`
- current deployable t100 policy: `Stage43-CI/CK floor`
- reason: Stage43-T is a small source-level TrajNet_crowds h100 split, while CK is the current global causal-only full-matrix audit and keeps the t100 floor.

## Next Required Actions

- Do not cite Stage43-T as global t100 success.
- If t100 is revisited, build a current-matrix-compatible source-family t100 gate with persisted feature names.
- Acquire or validate more h100 source support before making uniform t100 claims.
- Keep t100 as raw-frame diagnostic until source-stable causal evidence is positive and easy-safe.

## Boundary

- Dataset-local/raw-frame 2.5D only.
- No metric or seconds-level claim.
- No true 3D or foundation claim.
- No Stage5C execution.
- No SMC.

## Gate

| gate | passed |
| --- | --- |
| `stage43_s_precondition_present` | `True` |
| `stage43_t_local_positive_present` | `True` |
| `stage43_t_easy_safe_under_own_split` | `True` |
| `stage43_ck_global_t100_floor_confirmed` | `True` |
| `split_scope_difference_recorded` | `True` |
| `global_t100_not_overclaimed` | `True` |
| `uniform_t100_not_overclaimed` | `True` |
| `feature_contract_audited` | `True` |
| `no_denied_protocol_fragments` | `True` |
| `no_future_or_test_leakage` | `True` |
| `no_metric_seconds_stage5c_smc_claim` | `True` |
| `long_objective_kept_active` | `True` |
