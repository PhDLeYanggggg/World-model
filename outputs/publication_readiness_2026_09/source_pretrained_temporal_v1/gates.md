# Evidence Gates

This is a completed training-side representation experiment, not a new deployment
or the main 8-to-12 benchmark. Gates are scientific claims, not a count of completed files.

| Gate | Status | Evidence |
| --- | --- | --- |
| Fixed comparison registered before execution | Pass | Commit 06f97771; 36 heads, three arms, four sites, three seeds. |
| Actual neural training | Pass | 360,000 optimizer updates, all planned heads completed. Frozen ResNet is not end-to-end encoder training. |
| Role and row alignment | Pass within approved offline contract | 15,430 queries, 25,300 images, 36 matched sampling streams, future-target input poisoning and excluded-role checks. |
| Strict sensor-as-of observation | Not established | Supplied histories may use later annotation controls. This limitation remains explicit. |
| Reproducibility | Pass for local checked artifacts | 36 exact head replays, three exact encoder chunk replays, nine OOF archives, 339 artifacts unchanged on zero-update resume. |
| Actual source-held forecasting benefit | Fail | All 36 fits lose to stationary CV. Equal-site arm gains -0.070%, -1.908%, -6.102%. |
| Added pretrained appearance benefit | Fail in this comparison | Current minus geometry -1.838pp; sequence minus geometry -6.032pp. Both conditional intervals below zero. |
| Added temporal appearance benefit | Fail in this comparison | Sequence minus current -4.194pp, conditional interval [-6.229,-2.673]. |
| Easy preservation at the main 2% budget | Not established | This diagnostic easy subset has zero baseline error; percentage degradation is undefined, not a pass. Absolute harms are disclosed. |
| Independent scene calibration and confirmation | Not run | Four sites have been explored; shared-training-fold bootstrap is conditional development evidence only. |
| New policy or neural deployment | No | No learned intervention gate was trained or selected here. Oracle is not a policy. |
| Main/t+50/Stage37/external benchmark | Not run | Do not transplant source-subset scores to historical benchmark claims. |
| Publication candidate / general world-model contribution | Not established | Positive reliable forecasting, useful modalities and safe joint selection remain missing. |
| Stage5C execution | Forbidden / false | No execution. |
| SMC | Forbidden / false | Not enabled. |

Focused verification: 22 tests passed. The full legacy suite was not rerun; it
contains non-hermetic integration/training routines outside this unchanged scope.
Runtime: native arm64 Python 3.11, Torch 2.12.0, torchvision 0.27.0, NumPy 2.4.6;
four CPU threads, one inter-op thread, zero loader workers. No CREATE job submitted.

Result provenance: `fresh_run` extraction, fitting, analysis and verification;
`cached_verified` upstream source assets; `not_run` main and confirmation evaluation.
