# Stage43-BA Tail Adapter Source Blocker Audit

- source: `fresh_stage43_ba_tail_adapter_source_blocker_audit`
- result_source: `fresh_source_family_blocker_audit_from_stage43_p_and_az`
- verdict: `stage43_ba_tail_adapter_source_blocker_audit_pass`
- gate: `13 / 13`
- performance leader policy hash: `9155067aacf42bc8d8e67745c1cf5e05b729f95a88cf65d33d88b9a06c21484b`
- model hash: `03497313f878a1ec69fd7d2824842fee0acfa79c38dc9d667c6d6ac53ef4c331`

## Summary

- test sources: `4`
- positive switched sources: `2`
- safe-floor blocked sources: `2`
- catastrophic ungated blocked sources: `2`
- positive domains: `2 / 3`
- nonnegative domains: `3 / 3`
- uniform positive transfer claim allowed: `False`
- floor necessity supported for blocked sources: `True`

## Source Rows

| source family | source | rows | selected lift | ungated lift | switch | status | diagnosis |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `TrajNet_crowds` | `crowds_zara03.txt` | 9540 | `62.84%` | `61.92%` | `84.91%` | `positive_switched` |  |
| `UCY` | `obsmat.txt` | 70585 | `54.01%` | `53.45%` | `78.08%` | `positive_switched` |  |
| `TrajNet_biwi` | `biwi_hotel.txt` | 7685 | `0.00%` | `-1012.75%` | `0.00%` | `safe_floor_blocked` | floor_required_ungated_catastrophic_negative_transfer |
| `TrajNet_mot` | `PETS09-S2L1.txt` | 1926 | `0.00%` | `-506.58%` | `0.00%` | `safe_floor_blocked` | floor_required_ungated_catastrophic_negative_transfer |

## Validation Blockers By Family

| family | validation rows | allowed horizons | block reasons | h100 blocked |
| --- | ---: | --- | --- | --- |
| `ETH_UCY` | 16611 | `none` | `blocked_validation_nonpositive:4` | `True` |
| `TrajNet_biwi` | 459 | `none` | `blocked_insufficient_validation_support:3` | `False` |
| `TrajNet_crowds` | 37153 | `10,25,50` | `blocked_validation_easy_harm:1` | `True` |
| `UCY` | 47223 | `10,25,50` | `blocked_h100_global_validation_contract:1` | `True` |

## Interpretation

Stage43-P is a strong aggregate performance leader, but it is not a uniform-positive source-transfer result. The blocked TrajNet_biwi and TrajNet_mot sources remain safe because the policy falls back to the floor; their ungated full-waypoint transfer is strongly negative, so the floor is necessary rather than cosmetic.

## Next Required Actions

- Do not claim uniform positive source transfer for Stage43-P.
- Repair blocked TrajNet_biwi and TrajNet_mot with source-specific training only if validation support becomes safe.
- Keep the safety floor for blocked sources because ungated full-waypoint transfer is strongly negative.
- Separate source-family labels from coarse domain labels in future paper tables.

## Gate

| gate | passed |
| --- | --- |
| `stage43_p_and_az_passed` | `True` |
| `source_rows_audited` | `True` |
| `positive_and_blocked_sources_separated` | `True` |
| `blocked_sources_have_diagnosis` | `True` |
| `floor_necessity_for_blocked_sources` | `True` |
| `uniform_positive_transfer_not_overclaimed` | `True` |
| `nonnegative_domain_boundary_recorded` | `True` |
| `validation_blockers_mapped` | `True` |
| `next_actions_recorded` | `True` |
| `no_future_or_test_leakage` | `True` |
| `claim_boundary_not_overstated` | `True` |
| `stage5c_and_smc_false` | `True` |
| `long_objective_kept_active` | `True` |
