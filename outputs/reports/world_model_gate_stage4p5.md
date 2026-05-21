# Stage 4.5 World Model Gates

Passed: `4/8`
Stage 5 ready: `False`

| Gate | Status | Evidence | Explanation | Next Fix |
| --- | --- | --- | --- | --- |
| Unit / DT Gate | pass | `{"dt_median": 0.09999999999999432}` | dt/velocity/coordinate audit is internally consistent. | If fail: use dataset time, not dense frame id. |
| Causal Observation Gate | pass | `{"official_velocity_source": "causal_fd", "central_fd": "diagnostic_only"}` | Official benchmark uses past-only velocity. | Keep native/central separated from official metrics. |
| Baseline Sanity Gate | pass | `{"cv_FDE1": 0.00053, "identity_FDE1": 0.00053, "cv_FDE10": 0.01143, "identity_FDE10": 0.01143}` | Identity hand physics must not damage inertial motion. | Disable non-observed forces. |
| Learned Dynamics Gate | fail | `{"strongest_causal_baseline": "constant_turn_rate_velocity", "baseline_FDE100": 0.0482, "best_learned": "residual_over_constant_acceleration", "learned_FDE100": 1.14498}` | Learned residual must beat strongest causal baseline by 5%. | Train on multi-step real rollout targets and type-specific dynamics. |
| Physical Validity Gate | pass | `{"strongest_validity": 1.0, "learned_validity": 1.0}` | Learned model must not degrade physical validity. | Add validity penalties and real geometry. |
| Multi-step Gate | fail | `{"one_step_FDE100": 1.15487, "multi_step_FDE100": 1442.48877}` | Multi-step loss should improve long-horizon rollout. | Use true rollout training, not only one-step residuals. |
| SMC Gate | premature | `{"status": "premature", "reason": "deterministic learned model is not competitive with strongest causal baseline"}` | SMC is only meaningful after deterministic proposal is competitive. | Fix deterministic dynamics first. |
| Stage 5 Readiness Gate | fail | `{"t100_verified": true, "strongest": "constant_turn_rate_velocity", "best_learned": "residual_over_constant_acceleration"}` | Stage 5 requires learned model > strongest causal baseline plus validity and coverage. | Do not enter Stage 5 yet. |
