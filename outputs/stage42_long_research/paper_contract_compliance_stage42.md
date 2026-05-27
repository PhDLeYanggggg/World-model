# Stage42-IE Paper Contract Compliance

- source: `fresh_stage42_ie_paper_contract_compliance`
- generated_at_utc: `2026-05-27T22:10:53.779230+00:00`
- git_commit: `34e1c98`
- input_hash: `12d3d16762aff6a9f4755a3b2eb4795da682f6eb227b4c25eb963e7c4c412ad1`
- gate: `14 / 14`
- verdict: `stage42_ie_paper_contract_compliance_pass`

## Purpose

Stage42-IE applies the Stage42-ID claim contract to the actual paper package files. It is a compliance verifier, not new training, conversion, download, or evaluation.

## Summary

- contract_verdict: `stage42_id_paper_claim_contract_pass`
- contract_gate_passed: `True`
- paper_files_total: `9`
- paper_files_existing: `9`
- paper_files_with_dataset_local: `9`
- paper_files_with_raw_frame: `9`
- paper_files_with_2_5d: `9`
- paper_files_with_stage5c_boundary: `9`
- paper_files_with_smc_boundary: `9`
- unbounded_overclaim_hits: `0`
- supported_anchor_count: `5`
- supported_anchor_covered: `5`
- blocked_claim_count: `7`
- blocked_claims_covered_as_limitation: `7`
- new_training_or_conversion: `False`

## Paper File Compliance

| file | exists | raw-frame | dataset-local | 2.5D | Stage5C boundary | SMC boundary | overclaim hits |
| --- | --- | --- | --- | --- | --- | --- | ---: |
| `outputs/stage42_long_research/paper_outline_stage42.md` | `True` | `True` | `True` | `True` | `True` | `True` | 0 |
| `outputs/stage42_long_research/method_draft_stage42.md` | `True` | `True` | `True` | `True` | `True` | `True` | 0 |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | `True` | `True` | `True` | `True` | `True` | `True` | 0 |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | `True` | `True` | `True` | `True` | `True` | `True` | 0 |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | `True` | `True` | `True` | `True` | `True` | `True` | 0 |
| `outputs/stage42_long_research/model_card_stage42.md` | `True` | `True` | `True` | `True` | `True` | `True` | 0 |
| `outputs/stage42_long_research/data_card_stage42.md` | `True` | `True` | `True` | `True` | `True` | `True` | 0 |
| `outputs/stage42_long_research/reproducibility_stage42.md` | `True` | `True` | `True` | `True` | `True` | `True` | 0 |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | `True` | `True` | `True` | `True` | `True` | `True` | 0 |

## Supported Anchor Coverage

| anchor family | covered | anchors |
| --- | --- | --- |
| `stage26_sdd` | `True` | Stage26, SDD |
| `stage37_external` | `True` | Stage37, external |
| `m3w_neural_v1` | `True` | M3W-Neural v1, protected |
| `stage42_full_waypoint` | `True` | full-waypoint, group-consistency |
| `t100_raw_replay` | `True` | t100, raw-frame |

## Blocked Claim Handling

| claim | status | covered as limitation |
| --- | --- | --- |
| true 3D or foundation world model | `blocked` | `True` |
| global metric or seconds-level performance | `blocked` | `True` |
| Stage5C latent generative execution or SMC readiness | `blocked` | `True` |
| JEPA or Transformer as independent main contribution | `blocked_or_diagnostic` | `True` |
| scene/goal or neighbor/interaction as independent main contribution | `blocked_or_diagnostic` | `True` |
| new external converted/evaluated metric-time data from HZ/IA/IB | `blocked` | `True` |
| t100 seconds-level long-horizon prediction | `blocked` | `True` |
