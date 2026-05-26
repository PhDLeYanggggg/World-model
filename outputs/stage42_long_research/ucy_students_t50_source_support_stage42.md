# Stage42-BU UCY_students T50 Source-Support Audit

- source: `fresh_ucy_students_t50_source_support`
- generated_at_utc: `2026-05-26T15:20:36.357796+00:00`
- git_commit: `b530050`
- input_hash: `a543cd798b4bd28eaa7e860c2fd7ddbcb145964b1d33a96119f2deb2a3ef4ab9`
- gate: `14 / 14`
- verdict: `stage42_bu_ucy_students_t50_source_support_pass_blocker_narrowed`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-BU 只审计 UCY_students calibrated t50 source-family support，不训练神经模型。
- UCY_students03 已在 calibrated subset 中；本步骤检查本地 students001/002/003 是否提供独立 t50-capable support。
- alternate formats / duplicate copies 不能算新的 independent source。
- future endpoints 只作为 baseline error label，不作为 inference input。
- 不使用 central velocity，不使用 test endpoints 构建 goals，不使用 test metrics 调 threshold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Summary

- br_additional_sources_needed_before_bu: `2`
- local_candidates_audited: `9`
- local_paths_found: `9`
- independent_t50_capable_sources: `UCY_students01, UCY_students03`
- independent_t50_capable_source_count: `2`
- new_independent_t50_sources_found: `UCY_students01`
- additional_independent_t50_sources_still_needed: `1`
- source_cv_ready: `False`
- ucy_students_t50_support_repaired: `False`
- remaining_blocker: `UCY_students still lacks one independent t50-capable same-family source for train/val/holdout source-CV.`

## Candidate Audit

| source_id | independent_key | role | exists | t50 | t100 | strongest@50 | improvement_vs_cv@50 | counted independent? |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: |
| `UCY_students01_raw` | `UCY_students01` | `canonical_independent_candidate` | True | 6445 | 1949 | `damped_velocity_0p25` | 0.36076107959073506 | True |
| `UCY_students01_trajnet_duplicate` | `UCY_students01` | `alternate_short_format_duplicate` | True | 0 | 0 | `None` | None | False |
| `UCY_students03_obsmat_existing_calibrated` | `UCY_students03` | `existing_calibrated_source` | True | 6491 | 3413 | `damped_velocity_0p25` | 0.444351310314571 | True |
| `UCY_students03_raw_duplicate` | `UCY_students03` | `alternate_format_duplicate` | True | 2793 | 879 | `damped_velocity_0p25` | 0.3345092697133971 | False |
| `UCY_students02_trajnet_short` | `UCY_students02` | `short_local_candidate` | True | 0 | 0 | `None` | None | False |
| `UCY_students01_trajnet_train_duplicate` | `UCY_students01` | `duplicate_copy` | True | 0 | 0 | `None` | None | False |
| `UCY_students03_trajnet_train_duplicate` | `UCY_students03` | `duplicate_copy` | True | 0 | 0 | `None` | None | False |
| `UCY_students01_stage5b_duplicate` | `UCY_students01` | `duplicate_copy` | True | 0 | 0 | `None` | None | False |
| `UCY_students03_stage5b_duplicate` | `UCY_students03` | `duplicate_copy` | True | 0 | 0 | `None` | None | False |

## Interpretation

- BU narrows the UCY_students blocker: `students001` is a real additional t50-capable same-family source, while `students002` is locally present but too short for t50.
- `students003` alternate files and TrajNet/stage5b copies are duplicates, so they are not counted as independent support.
- With only `students001` and `students003` as independent t50-capable sources, train/validation/holdout source-CV is still not possible.
- This is a blocker-narrowing audit, not a positive t50 transfer result.
