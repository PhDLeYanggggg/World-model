# Stage42-CW Runtime Replay Paper Refresh

- source: `fresh_synthesis_from_stage42_cv_runtime_batch_replay`
- generated_at_utc: `2026-05-26T20:01:07.785046+00:00`
- git_commit: `26c9345`
- input_hash: `ab12fa776711d359502006775453586eefe6af2863274ca25e3d25af00c6ac62`
- gate: `25 / 25`
- verdict: `stage42_cw_runtime_replay_paper_refresh_pass`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 protected dataset-local / raw-frame 2.5D 多智能体 world-state candidate。
- Stage42-CW 是 paper/reproducibility refresh，不重新训练，不调 threshold。
- Stage42-CV 的 runtime batch replay 使用真实 common validation/test rows，而不是 toy smoke test。
- runtime replay 精确复现 Stage42-CQ guard 决策和 selected trajectory。
- future waypoints / endpoints 只作为 supervised labels 或 evaluation labels，不作为 inference input。
- t+50 / t+100 仍是 raw-frame horizon，不能写成 seconds-level。
- dataset-local/raw-frame 不能写成 global metric。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Runtime Replay Evidence

- policy_hash: `4af6536f86499d5b39efa535bb81978398586d65746bb983571b642af7c92d59`
- validation rows: `53256`
- test rows: `55528`
- validation decision exact replay: `True`
- test decision exact replay: `True`
- test selected_xy max abs diff: `0.0`
- test selected ADE max abs diff: `0.0`
- test selected FDE max abs diff: `0.0`

## Test Metrics Vs Endpoint-Linear ADE

- all: `1.77%`
- t50: `1.07%`
- t100 raw-frame diagnostic: `3.48%`
- hard/failure: `1.93%`
- easy degradation: `0.25%`
- switch rate: `16.96%`

## Joint Safety Vs Endpoint-Linear

- near_collision@0.02 delta: `-0.00%`
- near_collision@0.05 delta: `-0.06%`
- p05 min group distance delta: `-0.01%`
- jagged-rate delta: `0.00%`

## Paper File Status

| file | refreshed | exact replay | no metric boundary | Stage5C/SMC boundary |
| --- | ---: | ---: | ---: | ---: |
| `outputs/stage42_long_research/method_draft_stage42.md` | True | True | True | True |
| `outputs/stage42_long_research/experiment_tables_stage42.md` | True | True | True | True |
| `outputs/stage42_long_research/model_card_stage42.md` | True | True | True | True |
| `outputs/stage42_long_research/reproducibility_stage42.md` | True | True | True | True |
| `outputs/stage42_long_research/a_journal_gap_stage42.md` | True | True | True | True |

## Interpretation

- Stage42-CW moves Stage42-CV from an isolated runtime report into the paper/reproducibility package.
- The evidence is deployment-reproducibility evidence: frozen policy artifact, runtime API, and batch rows all replay exactly.
- This strengthens the protected policy claim but does not expand it into true 3D, foundation, metric, seconds-level, Stage5C, or SMC claims.
