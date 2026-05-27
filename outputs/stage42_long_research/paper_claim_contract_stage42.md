# Stage42-ID Paper Claim Contract

- source: `fresh_stage42_id_paper_claim_contract`
- generated_at_utc: `2026-05-27T21:50:58.264144+00:00`
- git_commit: `0e7bf13`
- input_hash: `6b56de19b903d58986b4a5aac6741a79600d77da6fa697afde622721487182e4`
- gate: `15 / 15`
- verdict: `stage42_id_paper_claim_contract_pass`

## Meaning

This is a paper-package claim contract over the current Stage42 evidence closure. It does not train, convert, download, or evaluate data.
It turns supported and blocked claims into explicit allowed language, forbidden language, deployment boundaries, and evidence pointers.

## Mandatory Caveats

- `dataset-local/raw-frame`
- `2.5D`
- `not metric/seconds-level`
- `Stage5C false`
- `SMC false`

## Contract

| claim | source status | paper role | allowed language | deployment boundary |
| --- | --- | --- | --- | --- |
| protected dataset-local/raw-frame 2.5D multi-agent world-state candidate | `supported` | main framing only with strict boundary | main framing only with strict boundary | claim-specific protected use only |
| Stage26 SDD cost-aware selector remains the SDD deployable baseline | `supported_cached_verified` | baseline and historical development evidence | SDD deployable baseline under pixel-space raw-frame evaluation. | deployable for SDD fallback/selector use only |
| Stage37 external t50 safe selector is deployable for dataset-local raw-frame external t50 transfer | `supported_cached_verified` | external safety floor / comparison baseline | External dataset-local raw-frame t50 safety floor / deployable selector baseline. | deployable external t50 safety floor under current dataset-local setup |
| M3W-Neural v1 is a protected neural world-state candidate, not ungated neural deployment | `supported_cached_verified` | protected neural candidate evidence | Protected neural world-state candidate; not ungated neural deployment. | protected deployment only with floor/safe-switch; not floor-free |
| Stage42 protected full-waypoint/group-consistency policies are current source-level world-state evidence | `supported` | main Stage42 world-state evidence | Protected full-waypoint/group-consistency 2.5D world-state evidence. | protected deployment only with floor/safe-switch; not floor-free |
| Stage42-HV provides row-level batch replay for the t100 easy-guard runtime policy | `supported_cached_verified` | runtime/replay evidence, raw-frame diagnostic only | Raw-frame t100 replay/runtime evidence only, not seconds-level long-horizon claim. | diagnostic raw-frame runtime replay |
| true 3D or foundation world model | `blocked` | blocked_or_limitation_only | Limitation/blocker: claim boundary explicitly false in module lock, linter, and replay artifacts | not deployable |
| global metric or seconds-level performance | `blocked` | blocked_or_limitation_only | Limitation/blocker: restricted metric/time/source terms ready candidates remain zero; calibration/source confirmation incomplete | not deployable |
| Stage5C latent generative execution or SMC readiness | `blocked` | blocked_or_limitation_only | Limitation/blocker: all current artifacts keep Stage5C false and SMC false | not deployable |
| JEPA or Transformer as independent main contribution | `blocked_or_diagnostic` | blocked_or_limitation_only | Limitation/blocker: module lock blocked modules: ['scene_goal', 'neighbor_interaction', 'JEPA', 'Transformer'] | not deployable |
| scene/goal or neighbor/interaction as independent main contribution | `blocked_or_diagnostic` | blocked_or_limitation_only | Limitation/blocker: Stage42-CJ/CK and context closure select baseline-family control / close current residual context protocol | not deployable |
| new external converted/evaluated metric-time data from HZ/IA/IB | `blocked` | blocked_or_limitation_only | Limitation/blocker: IB conversion_ready_targets=0; converted=0; evaluated=0 | not deployable |
| t100 seconds-level long-horizon prediction | `blocked` | blocked_or_limitation_only | Limitation/blocker: Stage42-HV t100 is exact row-level raw-frame replay, not verified seconds-level calibration | not deployable |

## Paper File Status

| file | exists | bytes | raw/dataset caveat | Stage5C/SMC boundary |
| --- | --- | ---: | --- | --- |
| `outputs/stage42_long_research/paper_outline_stage42.md` | `True` | 47729 | `True` | `True` |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | `True` | 63952 | `True` | `True` |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | `True` | 70602 | `True` | `True` |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | `True` | 50135 | `True` | `True` |
| `outputs/stage42_long_research/model_card_stage42.md` | `True` | 60809 | `True` | `True` |
| `outputs/stage42_long_research/data_card_stage42.md` | `True` | 49368 | `True` | `True` |
| `outputs/stage42_long_research/reproducibility_stage42.md` | `True` | 55155 | `True` | `True` |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | `True` | 92463 | `True` | `True` |

## Summary

- closure_verdict: `stage42_ic_current_claim_evidence_closure_pass`
- closure_gate_passed: `True`
- supported_claim_count: `6`
- blocked_claim_count: `7`
- contract_row_count: `13`
- paper_files_total: `8`
- paper_files_existing: `8`
- paper_files_with_claim_caveat: `8`
- paper_files_with_stage5c_smc_boundary: `8`
- metric_seconds_claim_allowed: `False`
- stage5c_executed: `False`
- smc_enabled: `False`
- new_training_or_conversion: `False`
