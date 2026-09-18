# Training-Only Deferral Gates

This experiment is complete; the M3W research goal is not complete.
Do not collapse engineering checks, training diagnostics and research gates
into a single success count.

| Check | Status | Evidence and limitation |
| --- | --- | --- |
| Fixed six-branch training budget | pass | 48,000 fresh updates, all three seeds and both objectives |
| Data-role and input contract | pass within experiment | Training rows only; no held/main forecasts or future-label input; historical lineage problems are not cleared |
| Exact baseline fallback | pass | Score <= 0 and unsupported context return exact stationary CV |
| Matched comparison | pass | Three sampling streams matched to both new arms and cached dense control |
| Reproducibility checks | pass | 24 exact replays; 290 unchanged artifacts on zero-update completed resume; 32 targeted tests |
| Expected-cost deployed-action training signal | fail | All three seeds emit only baseline, gain 0% |
| Cost-supervised training signal | pass, training only | All three seeds exceed CV and dense aggregate gain, with less absolute easy harm |
| Stronger candidate dynamics | not established | Raw proposal +0.18436% versus dense +0.18724%; final gated hard gain below dense |
| Original easy-preservation gate | not evaluable here | Zero-error baseline denominator; absolute harm is still positive |
| Uniform training-site benefit | fail | deathCircle negative in all three seeds |
| Independent risk calibration | not_run | No calibration role opened; sigmoid is not a calibrated probability |
| Held-source forecast improvement | not_run | Prohibited by this training-only registration |
| Independent main/generalization evidence | not_run | Main roles unchanged and sealed |
| New deployment or submission readiness | false | Training-only aggregate gain is insufficient |
| Stage5C execution | false | Not authorized or run |
| SMC | false | Not enabled |

No early snapshot or seed is selected retrospectively. The isolated training
repair does not establish the planned scene-level joint intervention mechanism.
Units remain annotation pixels/past-normalized coordinates and raw frames.
