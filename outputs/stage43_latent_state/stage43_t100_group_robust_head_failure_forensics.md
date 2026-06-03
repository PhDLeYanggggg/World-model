# Stage43-DB T100 Group-Robust Head Failure Forensics

- source: `fresh_stage43_db_t100_group_robust_head_failure_forensics`
- result_source: `fresh_forensics_from_verified_cz_da_reports`
- verdict: `stage43_db_t100_head_failure_forensics_complete_policy_distill_next`
- gate: `12 / 12`
- deploy on current heldout t100: `False`

## Aggregate Deltas

- DA minus CZ t100 mean: `-0.000463`
- DA minus CZ min-without-group mean: `-0.001300`
- DA minus CZ switch-rate mean: `-0.042067`
- DA validation-to-test min-without-group gap mean: `-0.001247`
- root cause count: `7`

## Root Causes

- `trained_head_underperforms_policy_only`: `True`
- `group_worst_case_not_preserved`: `True`
- `group_worst_case_gap_vs_cz`: `True`
- `under_switching_relative_to_cz`: `True`
- `validation_to_test_group_gap`: `True`
- `not_an_easy_safety_failure`: `True`
- `not_a_no_signal_failure`: `True`

## Per Seed

| seed | DA t100 | CZ t100 | delta | DA min-without | CZ min-without | DA switch | CZ switch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4323` | `0.001229` | `0.001645` | `-0.000416` | `-0.000222` | `0.000807` | `0.082000` | `0.097600` |
| `4331` | `0.001203` | `0.001962` | `-0.000759` | `-0.000171` | `0.001033` | `0.050300` | `0.120300` |
| `4337` | `0.001703` | `0.001917` | `-0.000214` | `-0.000520` | `0.001146` | `0.074900` | `0.115500` |

## Repair Hypotheses

- `high` / `triggered`: DA optimizes gain/harm/delta labels but not the actual CZ deployment policy. Next test: Train a policy-distilled admissibility head using CZ leave-group-out selected switches as teacher labels.
- `high` / `triggered`: The support penalty is too indirect to protect worst-case source/scene/domain groups. Next test: Add explicit worst-group validation loss or group DRO style batch objective, then select by min-without-group.
- `medium` / `triggered`: The trained head is more conservative than CZ and misses useful t100 switches. Next test: Distill high-confidence CZ switches while keeping harm/easy conformal guard.
- `medium` / `triggered`: Validation group support does not transfer cleanly to test groups. Next test: Use nested leave-one-source/scene validation and choose checkpoints by heldout-group transfer.

## Interpretation

- DA is not a no-signal failure: every seed remains t100-positive and easy-safe.
- DA is also not ready to replace CZ: it loses t100 mean, loses worst-group robustness, and switches less often than CZ.
- The next repair should train toward the actual CZ robust policy decisions and worst-group transfer objective, not just generic gain/harm/delta labels.
- Dataset-local/raw-frame 2.5D only; no metric/seconds, true-3D, foundation, Stage5C, or SMC claim.
