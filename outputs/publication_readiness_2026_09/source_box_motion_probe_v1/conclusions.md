# Observed Motion Probability Probes

Sixteen fixed logistic models completed; no threshold or model selection.
Same four explored source sites and 15,430 windows; no main or external scoring.
Source: fresh_run fitting, cached_verified source geometry, fresh coefficient replay.

| Target | Contrast (positive means improvement) | Motion minus quality | Conditional four-site 95% CI |
| --- | --- | ---: | --- |
| any_nonzero | brier | -0.0037178 | [-0.008898989135959795, 0.0014633364517297187] |
| any_nonzero | log_loss | -0.0115044 | [-0.02459673109243904, 0.0015878334581414522] |
| any_nonzero | auroc | -0.0052524 | [-0.013440082817013754, 0.0029352712064372732] |
| any_nonzero | auprc | -0.0117913 | [-0.028730445961737933, 0.005147816308967212] |
| over10_annotation_pixels | brier | -0.0015133 | [-0.0034354333203700325, -0.0004567700475068305] |
| over10_annotation_pixels | log_loss | -0.0051609 | [-0.012557679593562998, -0.0006327240195458383] |
| over10_annotation_pixels | auroc | +0.0124106 | [0.006674179309013789, 0.021013217190459232] |
| over10_annotation_pixels | auprc | +0.0024901 | [0.00022199190374124975, 0.005475099389621369] |

## Interpretation

The prespecified Brier contrast does not establish a motion-information improvement.
For the larger-excursion target, ranking improves modestly but Brier and log loss
worsen on every held site. Its motion AUROC ranges from 0.407 to 0.556, so the
positive difference partly improves a poor control rather than establishing a useful detector.
Both arms have worse larger-excursion Brier than the train-prevalence constant predictor
on all four sites. General nonzero-change discrimination does not improve consistently.
No safety threshold, calibration guarantee, trajectory gain or deployment follows.

## Numerical Boundary Disclosure

The fixed probe uses the stored float32 normalized future target restored to annotation
pixels. It yields 739 larger-excursion positives; the earlier exact raw-annotation
geometry audit yields 728. All 11 label differences are at a raw excursion of exactly
10 pixels, affected by normalization roundoff. Neither rows nor registered outcomes
are removed or silently changed. A post-hoc sensitivity readout of the same frozen
probabilities against exact raw labels is retained below. No refitting or selection.

| Site | Arm | Raw positives | Boundary differences | Raw-label Brier | Raw-label AUROC |
| --- | --- | ---: | ---: | ---: | ---: |
| coupa | quality | 137 | 0 | 0.0321673 | 0.398278 |
| coupa | motion | 137 | 0 | 0.0326611 | 0.407245 |
| deathCircle | quality | 204 | 11 | 0.0709026 | 0.473166 |
| deathCircle | motion | 204 | 11 | 0.0753157 | 0.497386 |
| gates | quality | 66 | 0 | 0.0412542 | 0.506503 |
| gates | motion | 66 | 0 | 0.0419780 | 0.512028 |
| hyang | quality | 321 | 0 | 0.0525262 | 0.545397 |
| hyang | motion | 321 | 0 | 0.0529458 | 0.555518 |

This threshold is an annotation-coordinate diagnostic, not a physical movement definition.
The raw-label readout is supplementary, not a replacement selected for better scores.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/probe_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_probe_v1.json
.venv-pytorch/bin/python scripts/probe_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_probe_v1.json --replay
.venv-pytorch/bin/python scripts/report_m3w_source_box_motion_probe.py
```

All models converged within the fixed budget; maximum 1462 iterations.
Summed fitting time 31.422s. Sixteen exact coefficient replays.
Completed resume adds zero fits; 34 artifacts unchanged. Private coefficients/predictions stay local.
The model is convex and fitted deterministically once per cell; no duplicated-seed evidence is claimed.
Two thousand site bootstrap resamples remain conditional, not independent confirmation.
Stage5C and SMC remain off. The M3W research goal remains unmet.
