# Frozen European Policy Family

- Purpose: model-selection generalization of gain/harm intervention over fixed
  causal motion forecasts. No new model was trained in this experiment.
- Family: nine four-source Torch forecasters; frozen motion-floor, old-controller,
  incumbent-relative and matched-reference heads; fixed ridge controls.36 dependent
  producer/controller/event/seed groups. All artifacts are locally hash-verified.
- Inputs: eight past positions, complete-history neighbor geometry, causal baseline
  and neural rollouts, motion-floor and incumbent decision bits. No future label,
  central velocity, test-endpoint goals or held-data normalization statistics.
- Output: choose the fixed protected-motion floor or fixed neural trajectory.
  Some arms override the incumbent; add-only preserves old neural interventions.
- Training lineage: producers and heads/preprocessing use only their registered
  source-training localities. None is fitted on the six model-selection localities.
- Original motion-floor schema355; matched380; incumbent-conditioned381. The new
  portable inference API accepts only geometry/history/origin and rejects extra
  label keys. Source replay is exact across all36 groups.
- Selection: none in this readout. No best-seed choice, threshold refit, calibration
  fit, deployment replacement, Stage5C or SMC.
- Limitations: mean gains are small; the full family fails worst-locality easy
  preservation. Easy-target branches do not uniformly beat classical baselines.
  Learned risk remains uncalibrated, and raw neural outputs are unsafe to promote.
- Evidence role: exploratory model selection, not independent final confirmation.
  Historical Stage37 evidence is not retroactively rehabilitated.
- Claims: detector-track image pixels and raw annotation steps only. No physical
  safety, metric/time calibration, true3D, generative-JEPA or foundation claim.
