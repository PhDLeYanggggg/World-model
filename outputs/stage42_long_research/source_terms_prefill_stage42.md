# Stage42-GB Source Terms Prefill

- source: `fresh_stage42_gb_source_terms_prefill`
- generated_at_utc: `2026-05-27T11:25:19.126490+00:00`
- git_commit: `bdb2254`
- input_hash: `74de99920d18292c0c1cfca38ae15d2b62439af4beb5c9bf0644d44c7b733b21`
- gate: `15 / 15`
- verdict: `stage42_gb_source_terms_prefill_pass`

## Role

- This stage turns the Stage42-GA local scan into a user-facing source-terms prefill draft.
- It does not accept terms, download data, convert data, train, evaluate, or mark any source as conversion-ready.
- The draft is intentionally not used as permission by the validator; the user must still edit/confirm the actual source terms template.

## Summary

- datasets_prefilled: `5`
- datasets_with_suggested_local_path: `5`
- raw_source_candidate_rows: `5`
- conversion_ready_now: `0`
- highest_priority_next_action: `FW-TERMS-ucy_crowd_original`

## Prefill Rows

| dataset | suggested local path | raw candidate? | next action |
| --- | --- | ---: | --- |
| `ucy_crowd_original` | `external_data/OpenTraj/datasets/UCY` | True | User verifies official terms, copies/edits suggested fields into source_terms_confirmation_template_stage42.json, then reruns the terms validator and guarded conversion queue. |
| `eth_biwi_original` | `external_data/OpenTraj/datasets/ETH` | True | User verifies official terms, copies/edits suggested fields into source_terms_confirmation_template_stage42.json, then reruns the terms validator and guarded conversion queue. |
| `trajnetplusplus_official` | `external_data/OpenTraj/datasets/TrajNet` | True | User verifies official terms, copies/edits suggested fields into source_terms_confirmation_template_stage42.json, then reruns the terms validator and guarded conversion queue. |
| `opentraj_toolkit` | `external_data/OpenTraj` | True | User verifies official terms, copies/edits suggested fields into source_terms_confirmation_template_stage42.json, then reruns the terms validator and guarded conversion queue. |
| `aerialmpt_or_other_topdown` | `data/aerialmpt/DLR_AerialMPT_Dataset.zip` | True | User verifies official terms, copies/edits suggested fields into source_terms_confirmation_template_stage42.json, then reruns the terms validator and guarded conversion queue. |

## Claim Boundary

- Prefill is not legal permission and not source conversion readiness.
- Local files remain insufficient for new benchmark claims without official terms/source identity confirmation and guarded conversion/no-leakage/source-CV.
- Current M3W remains protected dataset-local/raw-frame 2.5D; no true 3D, foundation, metric, seconds-level, Stage5C, or SMC claim.
