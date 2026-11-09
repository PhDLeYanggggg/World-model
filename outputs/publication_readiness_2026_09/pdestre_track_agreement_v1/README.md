# P-DESTRE: agreement does not select better tracks at equal count

The [camera study](../pdestre_camera_support_v1/README.md) recovered
70 complete RAPiD matches after the box-size interpretation check,
but not one fully scorable online roster. I tested two distinct uses
of a second detector: selecting which online tracks to keep, and
averaging observed positions for the same already-selected tracks.

## Fixed comparison

The original 11 windows, decoded images, RAPiD and MobileNet
boxes, and both online identity streams stay fixed. A RAPiD track
passes the new selector when it has a one-to-one IoU >=0.50 match to
the same MobileNet track in at least six of its eight observations,
including the last one. Controls keep exactly the same number per
window, by mean RAPiD confidence or by the number of framewise
matches regardless of the partner's ID. All three selections are
frozen before reading the scoring labels for this replay.

The evaluation uses the width/height diagnostic from the previous
study, not an official correction to the release. I match all
original RAPiD detections to reference labels first, then look up
those same matches for each retained subset. Removing tracks cannot
make the assignment step give them different reference identities.
All 108 original online histories are accounted for; no unknown
future becomes a label or a zero error.

## Selecting tracks

| Rule | Retained | Complete futures | No scored future | Windows with a complete pair | Fully scored retained subset |
|---|---:|---:|---:|---:|---:|
| All RAPiD tracks | 108 | 70 | 38 | 10/11 | 0/11 |
| Mean confidence, equal count | 79 | 56 | 23 | 9/11 | 0/11 |
| Frame agreement, equal count | 79 | 53 | 26 | 9/11 | 0/11 |
| Same-partner agreement | 79 | 52 | 27 | 8/11 | 1/11 |

The one fully scored subset has six of its original nine online
tracks, not the whole scene. It drops two complete tracks and one
without a scored future. Overall, same-partner agreement drops 18
of the 70 complete targets, versus 14 for the confidence control.
It does not beat the simple selector at equal count. I would not
use it as a hard admission rule. Unsupported trajectories are not
proven false detections.

## Averaging observed centers, no change to selected tracks

For the same-partner selection, average the two camera-box centers
where that partner is matched in the past, and retain the RAPiD
center on other observation steps. Fit the same anchored OLS4
predictor to the resulting eight-position history. This is a
measurement ablation, not a different forecasting model or a
second camera view. The original and averaged histories are
scored on exactly the same 52 complete matched trajectories.

| History used by OLS4 | Individual ADE (px) | Individual FDE (px) | Pair-relative ADE (px) |
|---|---:|---:|---:|
| RAPiD centers | 25.138 | 40.766 | 28.656 |
| Average matched centers | 24.852 | 40.448 | 25.549 |

Pair-relative ADE is the mean error of the predicted two-agent
relative position, over the 12 native steps. Its 10.84% reduction
is calculated across 155 complete pairs in eight windows. Those
pairs share agents and come from one linked source, so they are
not 155 independent samples. Five of eight windows improve. With
equal window weight, the comparison is 27.443 to 24.477 px; dropping
any one window leaves a pair-weighted change between -4.753 and
-2.380 px. This is a sensitivity calculation, not a confidence
interval. Individual improvements are small. The change does not
make an unscored target scorable, establish physical safety, or
show a neural policy gain.

Keep this as evidence that observed localization can change a
relative-trajectory metric without changing the motion model.
For the main paper, compare joint and independent forecasting on
fixed measured inputs and separate camera quality from policy
effects. This source cannot supply independent whole-scene
confirmation, and the original 2% checks were not run or changed.

## Replay

```bash
export SSL_CERT_FILE=/etc/ssl/cert.pem OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_pdestre_track_agreement.py freeze
.venv-pytorch/bin/python scripts/run_m3w_pdestre_track_agreement.py score
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_pdestre*.py tests/test_m3w_external_causal_sources.py tests/test_m3w_kit_ais.py tests/test_m3w_topdown_source_support.py
```

The replay reuses past camera receipts and retrieves only the
approved annotation ranges for this development study. Scoring
took 6.41 s and less than 379 MB maximum RSS on local arm64.
24 focused tests passed. No new detector fit or video inference,
CREATE job, reserved confirmation access, deployment, manuscript
edit, Stage5C or SMC.

[Registration](registration.json) | [Freeze](freeze.json) |
[Readout](analysis.json)
