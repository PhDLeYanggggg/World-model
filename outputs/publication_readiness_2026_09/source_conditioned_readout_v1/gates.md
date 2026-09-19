# Evidence Gates

Engineering checks and research claims are deliberately separate.

| Engineering check | Status | Evidence |
| --- | --- | --- |
| Fixed pre-fit registration | Pass | 5aa189b1, before the included pilot |
| Planned training budget | Pass | 24 fresh heads, 240,000 updates |
| Native Torch and recoverable execution | Pass | arm64 CPU4/inter-op1/workers0, atomic200-step checkpoints |
| Preserved forecast class and bounds | Pass | Unit equivalence tests; full prediction-bound checks |
| Matched sampling and loss | Pass | 24 regenerated streams/control matches and identical importance factors |
| Training roles/future-input checks | Pass within offline annotation protocol | Guarded training complements and future-label poison checks |
| Replay and resume | Pass | 24 exact forecast replays, zero-update completed resume |

| Research claim | Status | Reason |
| --- | --- | --- |
| Numerical conditioning improved | Supported | Logged clipping0%, much less static jitter than matched controls |
| Predictive gain over stationary CV | Fail | Both equal-site gains and all24held fits are negative |
| Useful hard/failure dynamics | Fail | Only negligible mixed-sign slice changes; no meaningful improvement |
| Visual representation lift | Fail | Centered appearance remains worse than geometry |
| Percentage easy-preservation gate | Not established | Static baseline error is zero; percentage is undefined |
| Independent generalization/calibration | Not run | Four already explored source sites/shared training folds only |
| New deployment | No | Optimization success is not forecasting success |
| Submission-ready method | No | Independent positive evidence and joint-intervention contribution still missing |
| Stage5C execution / SMC | False / False | Neither executed nor enabled |

Scope: offline supplied-annotation 8/12 steps at stride12 rawframes, pixel
coordinates. Not sensor-as-of, seconds, metric, true3D or a foundation model.
Historical external scores remain exploratory. No main/outer/external readout
is performed here. Do not add engineering passes to negative research gates to
claim a completed world model.
