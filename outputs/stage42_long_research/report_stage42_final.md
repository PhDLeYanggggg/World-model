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

## Stage42-H Addendum

- source: `fresh_run`
- report: `outputs/stage42_long_research/sequence_ablation_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_h_gate.md`
- verdict: `stage42_h_sequence_ablation_pass`

Stage42-H directly addresses the Stage42-G flattened-history negative result by training a causal temporal sequence encoder over past-only history windows. The result is strongly positive for history under a sequence model: removing history tokens reduces t+50 improvement by `0.4578` and hard/failure improvement by `0.4708` relative to the full sequence model. This means the project should not conclude that history is useless; the correct conclusion is that flattened history plus ridge selection was too weak to express the temporal signal.

The strongest raw sequence variant without the safe switch has better family-FDE than the safe-switch variant, but it is not deployment-ready because it has not passed the proximity/collision/floor-safety study required for removing the Stage37/teacher floor. Stage42-H therefore improves the causal evidence for temporal dynamics, while leaving the larger Stage42 gaps open: JEPA/full Transformer/full-waypoint-shape retraining, metric/time calibration, additional external datasets, and a floor-free safety mechanism are still not complete.

## Stage42-I Addendum

- source: `fresh_run`
- report: `outputs/stage42_long_research/sequence_full_waypoint_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_i_gate.md`
- verdict: `stage42_i_sequence_full_waypoint_partial`

Stage42-I connected the Stage42-H causal sequence encoder to reconstructed full-waypoint ADE/FDE labels. It trained four variants across three seeds each: full, no-history, no-neighbor, and no-static-context. The result is useful but not a pass for the full model: `sequence_waypoint_full` has negative protected ADE all/t50/hard (`-0.0106`, `-0.0321`, `-0.0116`) while preserving easy cases. History contribution is measurable but small on full-waypoint ADE/FDE (`t50 ADE delta = 0.0040`, `t50 FDE delta = 0.0094`).

The important positive signal is diagnostic: `sequence_waypoint_no_static_context` is positive on ADE all/t50/hard (`0.0115`, `0.0199`, `0.0129`) and FDE t50 (`0.0611`) with easy degradation `0.0`. This suggests the next full-waypoint repair should keep causal sequence history but add static/context gating or static dropout rather than mixing all static/context features unconditionally. Stage42-I therefore reduces the full-waypoint gap, but it does not make the full sequence-to-waypoint model deployable yet.

## Stage42-J Addendum

- source: `cached_verified_checkpoints_fresh_static_gate_eval`
- report: `outputs/stage42_long_research/static_gated_full_waypoint_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_j_gate.md`
- verdict: `stage42_j_static_gated_full_waypoint_pass`

Stage42-J repairs the Stage42-I static/context failure mode by using cached-verified Stage42-I full/no-static checkpoints as experts and selecting static mix weights on validation by domain/horizon. Test is evaluated once. The static-gated policy is positive on full-waypoint ADE all/t50/hard (`0.0362`, `0.0369`, `0.0397`), t+100 raw-frame diagnostic ADE (`0.0267`), FDE all/t50 (`0.0633`, `0.1166`), and preserves easy cases (`0.0` degradation).

This strengthens the full-waypoint world-state evidence: the problem was not simply that static context is useless; the problem was forcing static/context globally. Partial static experts (`alpha=0.25` and `0.50`) are useful when validation says they are safe, while full static remains harmful. The boundary remains important: Stage42-J is a fresh gate/eval over cached Stage42-I checkpoints, not a fresh checkpoint training run, and all claims remain dataset-local raw-frame 2.5D.

## Stage42-K Addendum

- source: `fresh_run`
- report: `outputs/stage42_long_research/fresh_static_gated_checkpoint_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_k_gate.md`
- verdict: `stage42_k_fresh_static_gated_checkpoint_pass`

Stage42-K trains the Stage42-J static-gating idea directly into a fresh `StaticGatedSequenceWaypoint` checkpoint with three seeds. It passes its fresh checkpoint gates (`9 / 9`) and improves over the failed Stage42-I full static+sequence head while preserving easy cases. The fresh checkpoint result is positive on ADE all (`0.0136`), t+100 raw-frame diagnostic ADE (`0.0159`), hard/failure ADE (`0.0148`), FDE all (`0.0312`), and FDE t50 (`0.0358`), with easy degradation `0.0`.

The honest boundary is that Stage42-K does not replace Stage42-J as the strongest full-waypoint static-gated evidence. Stage42-J's policy-level gate remains stronger on ADE all/t50/hard and FDE t50, and Stage42-K's ADE t50 mean is still negative (`-0.0122`). The result is therefore a useful fresh-checkpoint repair step, not a new best deployable model. Next work should make the learned static gate horizon-aware and explicitly repair the t+50 slice.

## Stage42-L Addendum

- source: `fresh_run`
- report: `outputs/stage42_long_research/horizon_static_gate_repair_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_l_gate.md`
- verdict: `stage42_l_horizon_static_gate_repair_pass`

Stage42-L directly targets the Stage42-K t+50 failure by adding horizon embeddings to the static gate, lowering t+50 static dropout/gate penalty, and using a t+50-weighted validation policy. It passes its gates (`11 / 11`) and repairs the t+50 ADE sign: Stage42-K had ADE t50 `-0.0122`, while Stage42-L reaches `+0.0020`. It also improves the fresh checkpoint on ADE all (`0.0219`), ADE hard/failure (`0.0240`), and FDE t50 (`0.0532`) while preserving easy cases (`0.0` degradation).

The boundary remains: Stage42-L is still weaker than the Stage42-J policy-level static gate (`0.0362` all, `0.0369` t50, `0.0397` hard). Therefore Stage42-L is the strongest fresh checkpoint in this static-gated branch so far, but Stage42-J remains the strongest static-gated full-waypoint evidence overall. All claims remain dataset-local raw-frame 2.5D; Stage5C and SMC remain disabled.

## Stage42-M Addendum

- source: `fresh_run`
- report: `outputs/stage42_long_research/policy_distilled_static_gate_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_m_gate.md`
- verdict: `stage42_m_policy_distilled_static_gate_partial`

Stage42-M attempts to distill the Stage42-J validation-selected domain/horizon static expert policy into a fresh checkpoint. It trains three seeds with domain and horizon embeddings and a static-gate teacher alpha derived from Stage42-J's selected experts. The teacher uses no test endpoints; future waypoints remain loss/eval labels only.

The result is a useful negative/partial result, not a deployment improvement. Stage42-M is positive on ADE all (`0.0161`), hard/failure (`0.0177`), and FDE t50 (`0.0729`) with easy degradation `0.0`, but ADE t50 is still negative (`-0.0015`) and it fails `10 / 12` gates. It improves FDE t50 over Stage42-L but loses on ADE all/t50/hard. The likely failure mode is that coarse domain/horizon alpha distillation increases static usage without teaching row-level gain/harm. Stage42-L remains the best fresh checkpoint in this branch; Stage42-J remains the strongest static-gated full-waypoint evidence overall.

## Stage42-N Addendum

- source: `fresh_run`
- report: `outputs/stage42_long_research/row_gain_static_gate_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_n_gate.md`
- verdict: `stage42_n_row_gain_static_gate_partial`

Stage42-N directly follows the Stage42-M failure by replacing slice-level alpha distillation with row-level train/val static gain, floor gain, harm, and switchability supervision. It also fixes the runtime rough edge from the first attempt by caching train/val row-teacher targets and writing heartbeats. The run is intentionally marked as a single-teacher-seed pilot, not a full teacher ensemble.

The result is mixed and must not be packaged as a t+50 success. It improves ADE all (`0.0250`) and ADE hard/failure (`0.0269`) over Stage42-L/M while preserving easy cases (`0.0` degradation), but ADE t50 becomes negative (`-0.0278`). Row-level teacher diagnostics show large static/switchable mass on t50 (`train t50 switchable 0.462`, `val t50 switchable 0.572`), yet the learned deployment policy still switches the wrong t50 rows. The new failure taxonomy is therefore sharper: alpha-style static-gate supervision is not enough, even when row-level; the next repair needs an explicit row-level gain/harm/switchability selector head or t50-specific teacher ensemble, not just a static gate target.

## Stage42-O Addendum

- source: `fresh_run`
- report: `outputs/stage42_long_research/explicit_gain_harm_selector_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_o_gate.md`
- verdict: `stage42_o_explicit_gain_harm_selector_partial`

Stage42-O tests the Stage42-N diagnosis by adding an explicit row-level selector head for switch probability, expected gain, harm probability, and uncertainty. It also fixes the strict evaluation protocol by using train-split feature normalization for train/val/test and adding a `no_test_statistics_normalization` gate. Future waypoints remain labels only, not inference inputs.

The strict result is useful but partial: ADE all improves to `0.0526`, hard/failure to `0.0535`, t+100 raw-frame diagnostic to `0.0602`, and easy degradation stays under the mean 2% gate at `0.0155`. However ADE t50 remains slightly negative at `-0.0008`, so Stage42-O must not be packaged as a t+50 success. It confirms that explicit gain/harm prediction is better than alpha-only distillation, but the next repair needs a t+50-specific teacher ensemble or per-domain horizon calibration.

## Stage42-P Addendum

- source: `fresh_run`
- report: `outputs/stage42_long_research/t50_gain_harm_selector_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_p_gate.md`
- verdict: `stage42_p_t50_gain_harm_selector_pass`

Stage42-P directly targets the Stage42-O t+50 failure by increasing t+50 train/val teacher weight and using a t+50-weighted validation policy search. It keeps the strict protocol from Stage42-O: train-only feature normalization, validation-only threshold selection, no test endpoint goals, no future waypoint input, and no central velocity.

The result repairs the mean ADE t50 sign while preserving all/hard/easy: ADE all `0.0515`, ADE t50 `0.0066`, ADE hard/failure `0.0533`, easy degradation `0.0086`, and FDE t50 `0.0574`, with gates `14 / 14`. The limitation is statistical: the 3-seed t50 CI low is still negative (`-0.0179`), so this is a gate-passing t+50 repair, not yet a paper-level stable t+50 claim. The next step is more seeds/bootstrap and combining Stage42-P's row gain/harm selector with Stage42-J's static expert policy.
