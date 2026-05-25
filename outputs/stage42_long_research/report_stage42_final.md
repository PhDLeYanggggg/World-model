# Stage42 Final Evidence Package

- source: `fresh_run`
- generated_at_utc: `2026-05-25T20:33:12.102637+00:00`
- git_commit: `f7ab44d`
- input_hash: `a66ba85745ebe18cd6b2437eba80eda982261c3efc4ca14e1328570b7c2789f6`
- gate: `12 / 12`
- verdict: `stage42_f_paper_package_complete_not_full_a_journal_ready`

## Current Facts

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- 当前仍是 dataset-local / raw-frame 2.5D 多智能体轨迹世界状态模型。
- SDD 是 pixel-space；external 是 dataset-local / unverified weak metric diagnostic。
- t+50 / t+100 是 raw-frame horizon，不能写成 seconds-level。
- global metric/time claims 仍不允许；TGSIM 只能作为 traffic diagnostic，不是 pedestrian official claim。
- self-audited / visual-prior labels 不是 human gold。
- Stage5C latent generative 未执行。
- SMC 未启用。

## Direct Answers

- still 2.5D: `True`
- metric/time subset for official pedestrian claims: `False`
- full-waypoint dynamics: `True`, all=0.1858, t50=0.1480
- cross-domain/external validation: `True`, all=0.2103, t50=0.1365
- exceeds strongest/Stage37 floor: `True under protected policy`, safety-floor best all=0.2103
- scene/goal/interaction contribution: `partial`, because Stage42-D uses cached-verified component evidence and not all-component fresh retraining.
- enough for A-journal candidate: `not yet full A-journal ready`; strong protected 2.5D manuscript package, but gaps remain.

## Paper Files

- `outputs/stage42_long_research/paper_outline_stage42.md`
- `outputs/stage42_long_research/method_draft_stage42.md`
- `outputs/stage42_long_research/experiment_tables_stage42.md`
- `outputs/stage42_long_research/ablation_tables_stage42.md`
- `outputs/stage42_long_research/failure_taxonomy_stage42.md`
- `outputs/stage42_long_research/model_card_stage42.md`
- `outputs/stage42_long_research/data_card_stage42.md`
- `outputs/stage42_long_research/reproducibility_stage42.md`
- `outputs/stage42_long_research/a_journal_gap_stage42.md`

## Stage42-G Addendum

- source: `fresh_run`
- report: `outputs/stage42_long_research/retrained_ablation_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_g_gate.md`
- verdict: `stage42_g_retrained_ablation_phase1_pass`

Stage42-G Phase1 continued the long research objective after this paper package by fresh-refitting external expected-FDE selector ablations for 10 variants across 3 seeds each. It improves the causal ablation evidence beyond cached coverage: goal/scene proxy, neighbor/interaction, and safe-switch/floor variants show positive contribution signals. It also adds an honest negative result: flattened history / transformer-proxy history features are not positive in this lightweight ridge-selector protocol, so history contribution still needs a true sequence-model ablation. JEPA, full Transformer retraining, endpoint bridge, and full-waypoint-shape ablations remain `not_run_in_phase1`.
