# Remaining TRAIN Case: Alternative Action Bank

Post-fit label-aware case diagnosis; not a new trained/deployed selector or a changed primary gate.
Fixed case: single1_seed43_controller2_dimensionless, eu-locality-082/51.
Rows: 78; known eligible easy rows: 3.
All six existing analytic baselines plus the frozen new neural candidate are reported.
Exact analytic equality to the floor is removed, not used as a fake safe intervention.

| Candidate | Real eligible | Known real easy | Individually safe easy | Minimum easy H/R | Label-aware feasible |
|---|---:|---:|---:|---:|---|
| constant_position | 74 | 3 | 0 | 13.618166410896555 | False |
| constant_velocity_causal_fd | 0 | 0 | 0 | None | False |
| damped_velocity_090 | 74 | 3 | 2 | 0.0 | True |
| damped_velocity_097 | 74 | 3 | 3 | 0.0 | True |
| history_ols4_velocity | 74 | 3 | 2 | 0.0 | True |
| history_ols8_velocity | 74 | 3 | 0 | 2.9663606276830126 | False |
| coverage_neural | 74 | 3 | 0 | 1.9310455230250796 | False |

Future labels are diagnosis only, never inference inputs. No per-row oracle is a real selector.
This action-bank probe does not rewrite the frozen148-view support diagnostic. Any new learned bank
would need prospective training/validation and independent calibration. Image pixels/raw frames only.
