# M3W-Neural v1

M3W-Neural v1 is a Stage41-protected neural world-dynamics candidate. It combines composite-tail bounded neural dynamics with a validation-selected safe-switch policy and the Stage37/teacher safety floor.

It is not true 3D, not metric, not seconds-level, not a foundation model, and not Stage5C/SMC.

## Files

- `README_GOAL_SUMMARY_M3W_NEURAL_V1.md` — detailed research ledger: attempted routes, failures, successes, current best deployable candidate, and remaining gaps.
- `report_m3w_neural_v1.md` — frozen result summary.
- `evidence_matrix_m3w_neural_v1.md/json` — gate and metric evidence.
- `selector_policy_m3w_neural_v1.json` — frozen policy metadata and hashes.
- `model_card_m3w_neural_v1.md` — intended use and limitations.
- `data_card_m3w_neural_v1.md` — dataset and leakage status.
- `reproducibility_m3w_neural_v1.md` — rerun commands.
- `paper_gap_m3w_neural_v1.md` — what is still missing before stronger publication claims.

Latest package inputs include the negative fixed-composer source-switch audits and the positive strict pure-UCY neural retrain/statistical evidence, so the frozen package records both the successful composite-tail path and the repaired source-only neural branch.

The package also includes the positive endpoint-to-full bridge audit: domain-local endpoint neural dynamics pass actual full-waypoint ADE/FDE, multi-agent, proximity, and smoothness gates on ETH_UCY and TrajNet through a linear waypoint bridge. This strengthens world-state evidence without claiming learned waypoint-shape dynamics.

The endpoint-to-full bridge now also has fresh 2000-bootstrap per-domain statistical support on ETH_UCY and TrajNet. The lower bounds are positive for all/t50/hard/multi-agent ADE and all/t50 FDE, but this is still protected linear-bridge evidence rather than ungated learned full-waypoint shape dynamics.

The required ablation coverage audit is now packaged. It covers no-history, no-neighbor, no-scene/goal, no-interaction, no-JEPA, no-Transformer, and no-fallback. The newer same-protocol neural architecture audit records that pure Transformer/no-JEPA, JEPA-only/no-Transformer, and JEPA+Transformer hybrid attempts were negative or fallback-only under the Stage41 external protocol.

The package includes a calibrated learned-shape meta-policy as well. It selects protected waypoint-shape residual sources on validation, evaluates test once, and remains positive on ETH_UCY and TrajNet. The learned-shape contribution is small and protected, not an ungated neural replacement.

## Stage42-A Data Calibration Follow-Up

Stage42 Long Research Mode has started with a fresh data/calibration audit:

- report: `outputs/stage42_long_research/data_calibration_stage42.md`
- gate: `outputs/stage42_long_research/stage42_stage_a_gate.md`
- user actions: `outputs/stage42_long_research/user_action_required_stage42.md`
- result: Stage42-A gates `7 / 7`

The audit confirms that existing local converted state is sufficient to proceed to Stage42-B external validation and Stage42-C full-waypoint dynamics. It also confirms that global metric and seconds-level claims remain disallowed.

## Stage42-B External Validation Follow-Up

Stage42-B rebuilt a source-level/fold stress protocol over the frozen external evaluation pool and reran the protected package comparisons:

- report: `outputs/stage42_long_research/external_validation_stage42.md`
- source split: `outputs/stage42_long_research/external_source_split_stage42.json`
- gate: `outputs/stage42_long_research/stage42_stage_b_gate.md`
- result: Stage42-B gates `10 / 10`

Key fresh-run result:

```text
frozen_eval_pool_rows = 66303
evaluated_rows = 55528
protected_M3W_all_ADE_improvement = 0.2103
protected_M3W_t50_ADE_improvement = 0.1365
protected_M3W_t100_raw_frame_diagnostic_ADE_improvement = 0.1469
protected_M3W_hard_failure_ADE_improvement = 0.2038
protected_M3W_easy_degradation = -0.1451
ungated_neural_all_ADE_improvement = 0.2966
ungated_neural_easy_degradation = 1.2459
verdict = stage42_b_external_validation_pass_protected_neural_not_ungated
```

This confirms the protected neural candidate under the Stage37/teacher floor, and it also confirms the safety failure of ungated neural endpoint dynamics. Negative easy degradation means no easy-case harm under the report's metric convention. The result remains dataset-local raw-frame 2.5D evidence only; it is not metric, seconds-level, true 3D, Stage5C, or SMC.
