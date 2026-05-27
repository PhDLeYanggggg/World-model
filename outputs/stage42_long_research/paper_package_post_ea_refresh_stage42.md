# Stage42-EB Post-EA Paper Evidence Refresh

- source: `fresh_paper_refresh_from_stage42_dy_dz_ea`
- generated_at_utc: `2026-05-27T01:26:04.304363+00:00`
- git_commit: `20f10a9`
- input_hash: `cb86bc7a45a5d7fbab239c3bc8a8e27177ef7119c3831c0f202f87a4842c702e`
- gate: `12 / 12`
- verdict: `stage42_eb_post_ea_paper_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-EB 是 post-EA paper package refresh，不重新训练，不调 threshold，不把 paper update 当模型成功。
- 本阶段同步 Stage42-DY/DZ/EA：loss-family blocker、explicit group-consistency repair、UCY+TrajNet dual-domain bootstrap evidence。
- future endpoints / waypoints 只作为 supervised/evaluation labels，不能作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Refresh Content

## Stage42-EB Post-EA Paper Evidence Refresh

- source: `fresh_paper_refresh_from_stage42_dy_dz_ea`
- role: synchronize paper-ready artifacts after explicit physical consistency and dual-domain bootstrap evidence.
- This is a paper-package update from fresh Stage42-DY/DZ/EA evidence, not new training and not a threshold search.

### What Changed After EA

- scalar loss-family promotion remains blocked: best `proximity_occupancy_loss` all/t50/hard `25.51%` / `22.14%` / `23.74%`.
- explicit group-consistency is source-level promoted: all/t50/t100 raw/hard `24.72%` / `22.36%` / `14.35%` / `23.89%`.
- group-consistency delta vs Stage42-AM all/hard: `0.14%` / `0.14%`.
- near@0.05 is repaired from `1.94%` to `1.38%` in the DY checkpoint.

### Dual-Domain Evidence

- positive safe domains: `2`.
- UCY all/t50/hard: `35.58%` / `22.72%` / `33.78%`.
- TrajNet all/t50/hard: `32.07%` / `28.18%` / `31.29%`.

### Bootstrap Evidence

- bootstrap_n: `2000`.
- global all/t50/hard CI: `[32.56%, 33.23%]` / `[26.53%, 27.44%]` / `[31.51%, 32.26%]`; easy degradation CI `[-32.96%, -31.28%]`.
- UCY all/t50/hard CI: `[34.70%, 36.49%]` / `[21.38%, 24.18%]` / `[32.84%, 34.76%]`.
- TrajNet all/t50/hard CI: `[31.72%, 32.41%]` / `[27.72%, 28.61%]` / `[30.90%, 31.66%]`.
- near@0.05 final-base delta CI: `[-0.86%, -0.67%]`.

### Updated Paper Claim Boundary

- Supported: protected source-level group-consistency full-waypoint dynamics with UCY+TrajNet bootstrap-backed raw-frame evidence.
- Supported: explicit physical/group-consistency as a source-level full-waypoint repair route.
- Not supported as main claims: scalar loss weighting, goal/scene context, and neighbor/interaction context under current protocols.
- Not supported: ungated full-waypoint deployment or global primary full-waypoint replacement.
- Still forbidden: true 3D, foundation model, global metric/seconds-level claims, Stage5C execution, and SMC readiness.

## Paper File Status

| file | refreshed | dual-domain bootstrap | loss blocker | group claim | non-claims |
| --- | ---: | ---: | ---: | ---: | ---: |
| `outputs/stage42_long_research/paper_outline_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/method_draft_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/ablation_tables_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/failure_taxonomy_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/model_card_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/data_card_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/reproducibility_stage42.md` | True | True | True | True | True |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | True | True | True | True | True |

## Interpretation

- Stage42-EB makes the dual-domain statistical evidence available to the paper package.
- It upgrades paper wording from single-report evidence to a coherent claim boundary across paper artifacts.
- It keeps loss-family, goal/scene, neighbor/interaction, ungated full-waypoint, metric/seconds, Stage5C, and SMC overclaims blocked.
