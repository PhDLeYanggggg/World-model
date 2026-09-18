# Evidence Gates

This table separates completed engineering work from supported scientific claims.
It is not a deployment gate count or the main 8-to-12 benchmark.

| Gate | Status | Evidence |
| --- | --- | --- |
| Fixed experiment before fitting | Pass | Registration commit 4b5dadd9; two arms, four sites, three seeds, 24 heads. |
| Real neural training | Pass | 240,000 updates, every head completed; frozen ResNet18 is not end-to-end encoder training. |
| Past-indexed input alignment | Pass within the approved offline protocol | 123,440 frame keys; no all-identical eight-frame sequence; all 15,430 windows supported. |
| Fixed-role and sampling checks | Pass for checked artifacts | Disjoint training/held/main roles, 24 exact matched sample streams, six OOF archives recomputed. |
| Input repair implementation | Pass | All 15,430 real transforms pass common-offset and batch-composition checks; missing-token behavior is tested. |
| Local reproducibility | Pass | 24 exact checkpoint replays; completed resume adds zero updates and preserves 83 artifacts. |
| Shared-appearance repair beats failing sequence control | Pass as a development contrast | +5.3401pp centered and +4.3458pp normalized-centered, conditional intervals above zero. |
| Actual forecasting beats stationary CV | Fail | New gains -0.7622% and -1.7564%; all 24 held fits negative. |
| Added visual input beats geometry-only | Fail | Both registered contrasts negative with conditional intervals below zero. |
| RMS amplification improves centered input | Fail | -0.9942pp, conditional interval [-1.7136, -0.4306]. |
| Easy preservation at the main 2% budget | Not established | Zero-error baseline makes percentage degradation undefined. Absolute harms 0.020320/0.041864 annotation pixels are disclosed. |
| Strict sensor-as-of observation | Not established | Supplied histories may use later interpolation controls. |
| Independent calibration or confirmation | Not run | Four explored sites/shared fitting folds; conditional development intervals only. |
| Main/t+50/Stage37/external benchmark | Not run | Source-subset results cannot replace those evaluations. |
| New deployment or submission-ready method | No | Reliable positive candidate dynamics and independent evidence remain missing. |
| Stage5C execution | Forbidden / false | Not executed. |
| SMC | Forbidden / false | Not enabled. |

All 32 scoped tests passed. The full legacy integration/training suite was not
rerun; no claim of repository-wide test success. Native arm64 CPU with four
compute threads, one inter-op thread and zero loader workers; no CREATE job
submitted. Training took 764.240 summed fitting seconds including the pilot;
the main log spans 772.059 seconds. All required processes have exited normally.

`fresh_run`: past-only audit, 24 heads, analysis and artifact verification.
`cached_verified`: upstream assets, frozen embeddings and 36 matched controls.
`not_run`: the new independent-event support investigation, main/outer scoring,
new intervention policy, external evaluation and independent confirmation.
