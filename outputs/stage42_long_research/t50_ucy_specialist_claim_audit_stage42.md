# Stage42-IL T50 UCY Specialist Claim Audit

- source: `fresh_stage42_il_t50_ucy_specialist_claim_audit`
- generated_at_utc: `2026-05-28T01:59:58.252127+00:00`
- input_hash: `c4ab3166d5570a016c59da6ef2bdf582f7e459f1a9599c16ff99423e421b577b`
- gate: `16 / 16`
- verdict: `stage42_il_ucy_specialist_claim_audit_pass`

## Purpose

Stage42-IK repaired the UCY fallback-only weak source in the Stage42-II/IJ t+50 ensemble. Stage42-IL turns that into a paper-safe evidence record: it measures the exact delta, verifies non-UCY rows remain unchanged, and states what claims are allowed or blocked.

## Delta Versus Stage42-II

| metric | delta |
| --- | ---: |
| ADE all | 0.037627 |
| ADE t50 | 0.023160 |
| ADE t100 raw diagnostic | 0.039687 |
| ADE hard/failure | 0.038954 |
| easy degradation | 0.000000 |
| switch rate | 0.086803 |

## UCY Weak Source Repair

| item | value |
| --- | ---: |
| rows before / after | 9540 / 9540 |
| t50 before | 0.000000 |
| t50 after | 0.122892 |
| t50 delta | 0.122892 |
| all after | 0.196091 |
| hard after | 0.207360 |
| easy degradation after | 0.000000 |

- non_ucy_max_abs_delta: `0.000000101979`

## Supported Claims

| claim | status | evidence |
| --- | --- | --- |
| Stage42-IK repairs the Stage42-II/IJ UCY fallback-only t50 weak source under a row-aligned source specialist. | `supported_fresh_composition_eval` | UCY t50 0.000000 -> 0.122892; alignment rows 9540 |
| The Stage42-II non-UCY ensemble decisions are unchanged by the IK composition. | `supported_fresh_audit` | max_abs_non_ucy_domain_metric_delta=0.000000101979 |
| All powered t50 source files are nonnegative/positive after IK. | `supported_fresh_audit` | positive_powered_t50_source_count=3/3 |
| IK improves the global Stage42-II ensemble while preserving easy cases. | `supported_fresh_audit` | all_delta=0.037627; t50_delta=0.023160; easy_degradation=0.000000 |

## Blocked Claims

| claim | status | reason |
| --- | --- | --- |
| IK proves a new independent external-domain generalization result. | `blocked` | IK is a source-specialist composition using cached-verified row-aligned UCY full-waypoint branch evidence. |
| IK is new training. | `blocked` | IK source labels mark new_training=not_run; it is a fresh composition/evaluation audit. |
| IK allows metric or seconds-level claims. | `blocked` | claim boundary remains dataset-local/raw-frame only. |
| IK permits Stage5C or SMC. | `blocked` | Stage5C and SMC remain false in all relevant artifacts. |

## Interpretation

- IK can be used as source-specialist composition evidence for repairing the UCY t50 weak source.
- IK should not be written as new independent-domain generalization, new training, metric/seconds calibration, Stage5C, or SMC.
- This strengthens the external validation ledger by removing a fallback-only powered source while keeping the claim boundary narrow.
