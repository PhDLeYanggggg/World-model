# Evidence Gates

| Gate | Status | Evidence |
| --- | --- | --- |
| Preregistration | Pass | b2276809 pushed before included pilot; all24heads and240kupdates fixed. |
| Original expected objective restored | Pass | Exact train-fold loss/gradient identities; no self-normalization or clipped factors. |
| Actual neural fitting | Complete | 24heads x10,000updates, no held-driven stopping or selection. |
| Training-only factors and role controls | Pass for checked path | Probabilities from training episode membership; held/main/outer role rejection and label-poison checks. |
| Exact exposure control | Pass | All24draw streams match the previous uncorrected controls. |
| Reproducibility | Pass locally | 24exactreplays,6OOFrecomputations,84artifacts preserved on zero-update resume. |
| Reduce previous sampler harm | Pass in development | +37.2947/+54.7171pp against the two uncorrected controls. |
| Beat original uniform controls | Partial harm reduction | Geometry contrast interval crosses0; centered improves a failing control. |
| Positive forecast gain over CV | Fail | -0.0321%/-0.2746%; all24heldfits negative. |
| Added visual contribution | Fail | Centered minus geometry -0.2425pp with conditional interval below0. |
| Hard/nonzero gain | Fail as a robust claim | Nonzero gains negative; hard-slice signs mixed or negative. |
| Easy degradation <=2% | Not established | Zero-error CV floor makes percentage undefined; absolute harms are reported. |
| Independent calibration/confirmation | Not run | Four explored source sites/shared fitting folds only. |
| Main/t+50/external comparison | Not run | No result from this subset is substituted for the main benchmark. |
| Deployment/publication readiness | No | No useful new candidate or independent method evidence. |
| Stage5C | False | Not executed. |
| SMC | False | Not enabled. |

37scopedtests passed; no full legacy-suite claim. Both coordinate unit and horizon
remain annotation-pixel/raw-frame or annotation-step descriptions, not metric
or seconds. Engineering passes are not counted as scientific success. The goal
remains active and unmet. Next training-side gradient diagnosis is not_run.
