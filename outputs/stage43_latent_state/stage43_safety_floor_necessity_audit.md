# Stage43-AJ Safety-Floor Necessity Audit

- source: `fresh_stage43_aj_safety_floor_necessity_audit`
- result_source: `fresh_audit_over_cached_verified_prior_fresh_evidence`
- gate: `10 / 10`
- verdict: `stage43_aj_safety_floor_necessity_confirmed`
- floor necessity confirmed: `True`

## Evidence Summary

- protected easy degradation: `0.00%` vs ungated `55.72%`
- protected t100 diagnostic: `-17.79%` vs ungated `-72.12%`
- no-baseline-floor t50 delta mean over `3` seeds: `12.83%`; positive seeds `3 / 3`
- raw-best scene variant `full_scene` t50 `36.09%` but easy `9.59%`
- safe-best scene variant `geometry_route` t50 `35.32%` and easy `0.00%`
- slice-safe policy t50 `37.16%`, easy `0.00%`, h100 floor rate `100.00%`

## Conclusion

- The floor is not removable globally in the current Stage43 evidence.
- The floor is a core safety mechanism, not merely a cosmetic crutch.
- Partial floor relaxation is supported only on validation-selected supported slices.
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
| global_floor_not_removable | True |
| no_future_or_test_leakage | True |
| no_metric_seconds_stage5c_smc_claim | True |
