# VisDrone-MOT: can better association recover the group?

I kept the 1,193 person boxes from the [camera pilot](../visdrone_mot_train_intake_v1/README.md)
and changed how they are linked. Every arm sees the same boxes, in the same
order, from the same eight observed frames. The assignment cost, matching gate,
track expiry and label scorer are identical. A projected box is used only for
assignment; a track still emits the actual detection.

I compared last-box association, two-observation velocity, background camera
compensation, and their combination. I estimated each camera step from only the
previous and current images, excluding person boxes, then used the identity
transform when the fit lacked support. This is a small motion ablation, not a
benchmark of [SORT](https://arxiv.org/abs/1602.00763) or
[BoT-SORT](https://arxiv.org/abs/2206.14651), which use different tracking models.

| Association | Eight-step online histories | Stable label identity | Complete 12-step future | Full online roster scored |
| --- | ---: | ---: | ---: | ---: |
| Last detected box | 53 | 17 | 14 | 0/4 |
| Two-observation velocity | 59 | 15 | 11 | 0/4 |
| Camera compensation | 62 | 24 | 21 | 0/4 |
| Camera compensation + velocity | 59 | 19 | 14 | 0/4 |

Camera compensation recovered seven additional complete histories. Six of the
seven came from the clip with the largest accepted image displacement, where
accepted steps moved the image center by about 11 to 13 pixels. The camera fit
was accepted on 19 of 28 transitions overall. It was rejected on every
transition in the clip where none of the arms recovered a complete match.
Adding a two-point velocity estimate did not help on these clips. The camera
arm also emitted more unmatched histories (38 versus 36), so the gain in
matched trajectories does not fix whole-roster coverage.

Of 76 complete pedestrian references, only 50 have even one geometrically
eligible detection at every past step, allowing duplicate use of a box and
ignoring identity. This is an upper bound, not an achievable tracking score.
It isolates a limit that no reassignment of these detections can remove under
the same scorer. The earlier 39/76 count uses unique per-frame box matches and
is therefore stricter than this bound.

The next input experiment should keep the camera arm fixed and improve box
coverage, rather than stacking more association rules. Compare both detection
coverage and whole-roster support before fitting a forecasting head. These
four reused training windows are a diagnostic; no forecast accuracy, risk
pass, or independent evaluation is reported. No validation or test data was
opened, and the original intervention risk rule is unchanged.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_association.py
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_association.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_visdrone_association.py tests/test_m3w_streaming_boxes.py tests/test_m3w_visdrone_camera_pilot.py
```

This run uses the previously cached JPEGs and detections. It replays the
original online IDs exactly and freezes the new assignments before it reads
training labels for scoring. [Registration](registration.json),
[per-frame camera fits and prefix hash](freeze.json), and
[per-clip support](readout.json).
