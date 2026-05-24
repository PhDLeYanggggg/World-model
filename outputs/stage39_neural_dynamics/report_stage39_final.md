# Stage39 Final Report

- 当前不是 true 3D world model。
- 当前不是 large-scale foundation world model。
- external coordinates remain dataset-local / unverified weak metric diagnostic.
- t+50/t+100 remain raw-frame horizons, not seconds-level claims.
- Stage5C executed: `False`
- SMC enabled: `False`

## Result

- deployment decision: `keep_stage37_selector`
- best neural: `Transformer_only`
- best neural metrics: `{'rows': 16000, 'all_improvement': 0.13200527968500975, 't10_improvement': 0.3025072845476737, 't25_improvement': 0.0, 't50_improvement': 0.08301228991324938, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1523895057047563, 'easy_degradation': 0.0006815189764808327, 'harm_over_fallback': -0.13887056636996567}`
- Stage37 same-subset metrics: `{'rows': 16000, 'all_improvement': 0.13200527968500975, 't10_improvement': 0.3025072845476737, 't25_improvement': 0.0, 't50_improvement': 0.08301228991324938, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1523895057047563, 'easy_degradation': 0.0006815189764808327, 'harm_over_fallback': -0.13887056636996567}`
- JEPA downstream lift: `{'source': 'fresh_run', 'failure_auroc_base': 0.9880898558164535, 'failure_auroc_with_jepa': 0.5595315549772284, 'failure_auroc_lift': -0.4285583008392251, 'failure_auprc_base': 0.9858232078403314, 'failure_auprc_with_jepa': 0.454838276741377}`
- external split repair: `{'UCY': 'available_heldout_test', 'ETH_UCY': 'not_run_blocker: available rows are train-only under frozen Stage37 split; rebuilding held-out test would invalidate frozen policy/test protocol', 'TrajNet': 'not_run_blocker: train/val rows exist but no frozen held-out test split; requires Stage40 split rebuild and retuning on val only', 'OpenTraj_mixed': 'not_run_blocker: mixed test currently UCY; non-UCY held-out requires new split'}`
- gates: `11 / 13`
- verdict: `stage39_neural_dynamics_diagnostic_keep_stage37`

## Interpretation

- Stage39 begins real neural dynamics training under Stage37 protection.
- Neural models are not deployed unless they beat Stage37 on all/t50/hard while preserving easy cases.
- If gates fail, Stage37 selector remains the current external best.
