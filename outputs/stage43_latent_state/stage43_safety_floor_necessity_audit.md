# Stage43-AJ Safety-Floor Necessity Audit

- source: `fresh_stage43_aj_safety_floor_necessity_audit`
- result_source: `fresh_audit_over_cached_verified_prior_fresh_evidence`
- gate: `13 / 13`
- verdict: `stage43_aj_safety_floor_necessity_confirmed`
- floor necessity confirmed: `True`

## Evidence Summary

- protected easy degradation: `0.00%` vs ungated `7.86%`
- protected t100 diagnostic: `-27.90%` vs ungated `-34.05%`
- no-baseline-floor t50 delta mean over `3` seeds: `11.12%`; positive seeds `2 / 3`
- raw-best scene variant `full_scene` t50 `36.09%` but easy `9.59%`
- safe-best scene variant `geometry_route` t50 `35.32%` and easy `0.00%`
- slice-safe policy t50 `37.16%`, easy `0.00%`, h100 floor rate `100.00%`
- self/conformal guard: ungated easy `55.72%` vs conformal easy `0.00%`, conformal t100 `0.00%`
- bounded residual safe relaxation: all `38.00%`, t50 `26.96%`, easy `0.00%`
- latest Stage43-P tail adapter: all `50.25%`, t50 `51.23%`, t100 `0.00%`, easy `0.00%`, switch `70.45%`

## Conclusion

- The floor is not removable globally in the current Stage43 evidence.
- The floor is a core safety mechanism, not merely a cosmetic crutch.
- Partial floor relaxation is supported only on validation-selected supported source/horizon or scene-proxy slices.
- The latest tail-horizon adapter remains a floor-protected policy: it improves t50/full-waypoint metrics but floors unsafe t100 switches.
- Ungated neural/full-scene switching remains unsafe, especially for easy cases and raw-frame t100.

## Boundary

- Dataset-local/raw-frame 2.5D only.
- Future labels are supervision/eval only.
- No metric/seconds claim, no Stage5C, no SMC.

## Gate

| gate | passed |
| --- | --- |
| safety_floor_replay_available | True |
| protected_vs_ungated_reported | True |
| protected_reduces_easy_harm_materially | True |
| protected_reduces_t100_harm | True |
| no_baseline_floor_t50_degradation_stable | True |
| unsafe_raw_scene_proxy_blocked | True |
| partial_floor_relaxation_supported | True |
| self_gate_conformal_still_floor_protected | True |
| bounded_residual_is_floor_protected_relaxation | True |
| latest_tail_policy_uses_safe_floor | True |
| global_floor_not_removable | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
