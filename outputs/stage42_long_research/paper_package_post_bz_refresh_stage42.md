# Stage42-CA Post-BZ Paper Package Refresh

- source: `fresh_synthesis_from_stage42_by_bz_artifacts`
- generated_at_utc: `2026-05-26T16:19:04.044436+00:00`
- git_commit: `48c147e`
- input_hash: `778f70aa5d5eddb3dec4fe60a8af84646ea573206ba83403e375d7ef8e7394a8`
- gate: `10 / 10`
- verdict: `stage42_ca_post_bz_paper_package_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CA 只是 paper package refresh，不重新训练，不调 threshold，不执行 Stage5C，不启用 SMC。
- Stage42-BZ 的 t50 bootstrap evidence 是 protected policy evidence，不是 floor-free neural deployment。
- test rows 只用于最终 reporting/bootstrap，不用于 policy selection。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。

## Evidence Rows

| item | status | paper use | evidence |
| --- | --- | --- | --- |
| Stage42-BY protected t50 floor-relaxability repair | `stage42_by_t50_floor_relaxability_repair_pass` | point-estimate protected t50 slice repair | repaired=TrajNet|50, UCY|50; global t50=28.97%; easy=-37.05%; not floor-free neural |
| Stage42-BZ protected t50 bootstrap evidence | `stage42_bz_t50_repair_statistical_evidence_pass` | bootstrap-backed t50 statistical evidence | target union t50 CI=[28.52%, 29.45%]; hard CI low=28.51%; easy CI high=-25.16%; n=3000 |
| Stage42-BZ slice TrajNet|50 | `ci_positive_and_easy_safe` | slice-level t50 evidence | rows=9198; t50 CI=[29.80%, 30.67%]; easy CI high=-27.61%; switch=95.26% |
| Stage42-BZ slice UCY|50 | `ci_positive_and_easy_safe` | slice-level t50 evidence | rows=2340; t50 CI=[23.02%, 26.08%]; easy CI high=-8.16%; switch=65.00% |

## Paper File Status

| file | refreshed | floor boundary | metric boundary |
| --- | ---: | ---: | ---: |
| `outputs/stage42_long_research/paper_outline_stage42.md` | True | True | True |
| `outputs/stage42_long_research/method_draft_stage42.md` | True | True | True |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | True | True | True |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | True | True | True |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | True | True | True |
| `outputs/stage42_long_research/model_card_stage42.md` | True | True | True |
| `outputs/stage42_long_research/data_card_stage42.md` | True | True | True |
| `outputs/stage42_long_research/reproducibility_stage42.md` | True | True | True |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | True | True | True |

## Interpretation

- Stage42-CA makes BY/BZ paper-package-visible evidence rather than a standalone loose report.
- It does not train, tune, or execute any latent generative / SMC component.
- It explicitly preserves the protected-policy-only claim boundary.
