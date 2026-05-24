# M3W Paper Package Draft

## Working Title

M3W: Real-World Multimodal Agent-Scene World Modeling for Top-Down Multi-Agent Forecasting

## Abstract Draft

We study real-world multimodal agent-scene world modeling for top-down multi-agent trajectory prediction under strict causal and leakage-free evaluation. The proposed M3W track combines JEPA-style latent representation learning with spatiotemporal Transformer dynamics and downstream heads for trajectory, baseline selection, failure prediction, interaction risk, occupancy, and physical-validity diagnostics. Experiments are conducted on SDD pixel-space raw-frame horizons with strongest causal baselines, BPSG-MA v1, Stage26 cost-aware selector, JEPA-only, Transformer-only, and hybrid variants. Current local-small results show that the Stage26 selector remains stronger than M3W hybrid; therefore M3W is not yet a submission-ready method, but the runtime, evaluation harness, gates, and failure analysis establish the next research loop.

## Method Summary

- Token schema: `agent_state`, `agent_history`, `scene_patch`, `scene_sdf`, `goal_region`, `interaction_edge`, `baseline_rollout`, `horizon`, `dataset`, `time`, `mask`.
- Data roles: `official_eval`, `supervised_training`, `representation_pretraining`, `simulation_curriculum`, `diagnostic_only`.
- Model families:
  - JEPA-only: latent prediction, no pixel reconstruction, no next-token objective.
  - Transformer-only: causal feature tokens for spatiotemporal dynamics.
  - Hybrid: JEPA latent plus Transformer dynamics with downstream heads.
- Heads: expected-FDE/selector, failure predictor, goal diagnostics, interaction risk, occupancy, physical validity.
- Correction head remains disabled because gates do not allow it.

## Experiment Summary

- Dataset: SDD official pixel-space benchmark derived from Stage24/26 feature store.
- Horizon: raw annotation-frame t+50 official; t+100 raw-frame diagnostic only.
- Metric status: pixel-space only; no verified homography/scale/effective seconds.
- Baselines: strongest causal baseline, BPSG-MA v1 fallback, Stage26 selector, JEPA-only, Transformer-only, Hybrid.
- Current best deployable: Stage26 selector, not M3W.

## Key Results

| model | t+50 improvement | hard/failure improvement | easy degradation | deployable |
| --- | ---: | ---: | ---: | --- |
| Stage26 selector | 0.1458365584 | 0.1123216763 | 0.0180883628 | yes |
| M3W torch small optimized | 0.0959744761 | 0.0702400714 | 0.0202588776 | no |
| M3W torch hard-focus | 0.0797643166 | 0.0459157386 | 0.0081676650 | no |

## Ablation Requirements Not Yet Satisfied

- JEPA downstream lift is not proven.
- Scene/goal contribution is not proven.
- Interaction contribution is limited to risk-head signal, not trajectory improvement.
- No simulation curriculum contribution is proven.
- No multi-seed/bootstrap CI is completed yet.

## Limitations

- Current model is not true 3D and not metric.
- SDD horizons are raw annotation-frame horizons, not seconds-level horizons.
- Scene/goal labels are self-audited or visual-prior only, not human gold.
- Current JEPA latent collapses or does not produce downstream lift in local-small.
- Hybrid model does not beat Stage26.

## Reproducibility Checklist

- No future endpoint input.
- No central velocity official input.
- No test endpoint goal construction.
- DataLoader multiprocessing disabled.
- PyTorch runtime requires sequential CPU environment variables listed in `optimization_report.md`.
- Checkpoints are written under `outputs/m3w/checkpoints/` and are not committed.

## Submission Readiness

Current status: not CCF-A submission candidate yet.

Required before submission:

1. Demonstrate JEPA or Transformer downstream lift over Stage26.
2. Pass t+50 or hard/failure gates while preserving easy cases.
3. Add ablations and bootstrap/multi-seed confidence intervals.
