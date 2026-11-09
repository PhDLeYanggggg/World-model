# VisDrone-MOT: more boxes do not yield more complete tracks

I fixed the [camera motion and association](../visdrone_association_v1/README.md)
and changed only how the same 32 images were passed to the COCO-pretrained
Faster R-CNN MobileNet detector. I compared the original 4-by-3 grid, an
8-by-5 grid, and pooling both grids with cross-grid nonmaximum suppression.
All three used 20% overlap, score threshold 0.35, and NMS IoU 0.5. This is a
fixed-grid diagnostic, not a benchmark of [SAHI](https://arxiv.org/abs/2202.06934).

| Detector input | Person boxes | References with an eligible box at all 8 steps, out of 76 | Online histories | Complete matches | Unmatched histories | Fully scored rosters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-by-3 grid | 1,193 | 50 | 62 | 21 | 38 | 0/4 |
| 8-by-5 grid | 1,629 | 48 | 86 | 24 | 59 | 0/4 |
| Both grids | 2,051 | 65 | 108 | 21 | 85 | 0/4 |

The coarse boxes reproduce the previous detector output exactly, in order;
the coarse tracking counts also replay for each clip. The eligible-box
count ignores identity and lets several references claim the same box. It
is a ceiling, not a track score. With unique per-frame assignments but no
identity requirement, the corresponding counts are 39, 41, and 55.

The grids see different pedestrians. The pooled ceiling improves by 15,
including 13 in the busiest clip. But pooled boxes introduce many new tracks
and do not improve complete stable matches. The fine grid alone recovers
three more complete matches but adds 21 unmatched histories. Unmatched does
not mean false detection: strict pedestrian labels can be missing, a box can
fail the geometric match, or a track can change identity.

**Next test:** keep coarse boxes as track starters and allow unmatched fine
boxes to extend an already supported track, without competing for a coarse
match. The fixed-grid runs do not justify increasing detector compute on the
entire dataset. These four reused training windows score input support, not
forecast accuracy or safety. No validation or test payload was read, no
new data roles were assigned, and the main manuscript is unchanged.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_detector_tiles.py
.venv-pytorch/bin/python scripts/audit_m3w_visdrone_detector_tiles.py --verify
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_visdrone_detector_tiles.py tests/test_m3w_visdrone_association.py
```

Inference used the local arm64 CPU, four Torch threads, and no loader
workers. It took about four minutes for both grids. Each image has a
hashed checkpoint and the command resumes incomplete work from it.
`--verify` checks the cached detections and recomputes tracking and support;
add `--recompute` to rerun detector inference as well. The aggregate
results replayed exactly. [Registration](registration.json),
[frozen checkpoint hashes](freeze.json), and [per-clip counts](readout.json).
