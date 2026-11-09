# Camera tracks and annotation coordinates on SDD

I ran three detectors on the same 736 observation frames from 23 SDD
training clips in five scenes. I wanted to know whether the camera
histories could support gain and harm learning, without using external
test labels or relying on hand-corrected histories at inference.

## The video and annotation rasters were different

Comparing boxes directly put 3,045 of 7,693 observed annotation
centers outside the video raster. I checked the size conversion with
background landmarks and applied the reference-image-to-video resize
to the annotation boxes, not the predictions. No forecast, detector,
or sampled window changed. The resize left zero observed centers
outside the raster. This is a conversion between two pixel rasters,
not a homography, metric calibration, or time calibration.

| Detector | Online histories | Stable matches before repair | Stable matches with all 12 outcomes after repair | Clips meeting development support |
| --- | ---: | ---: | ---: | ---: |
| Whole-image RAPiD | 16 | 0 | 1 | 0/23 |
| Tiled MobileNet | 44 | 3 | 2 | 0/23 |
| Tiled VisDrone-pretrained YOLO26n | 126 | 16 | 46 | 1/23 |

The original unscaled matches are retained as debugging records,
not performance evidence. The corrected results are the
[raster registration](raster_registration/registration.json) and
[readout](raster_registration/readout.json). The support floor is at
least eight complete matched histories across at least three windows
within a clip. The aerial detector clears it only on `hyang/video5`.
On the same five selected windows, tiling RAPiD without changing its
weights recovers one complete match; the whole-image input recovers
none. Most of the aerial detector's 552 available future positions are
flagged as generated in the release: nine are non-generated. I do not
treat those flags as independent label validation.

## Does the gain survive when unmatched histories are included?

I verified the original past-only input files byte for byte and
reattached corrected outcomes. Unmatched histories stay in the
denominator of their queries with missing outcomes. On the aerial
arm, 80 of 126 histories have no matched outcome. The paired utility
calculation subtracts forecast disagreement on missing steps and
normalizes by observed constant-velocity error. It does not impute an
error for an unmatched track.

| Aerial motion comparison | Known-step gain (pp) | Full-roster lower bound (pp) |
| --- | ---: | ---: |
| OLS4 vs. constant velocity | +4.99 | -43.17 |
| OLS8 vs. constant velocity | +6.82 | -44.12 |
| OLS8 vs. OLS4 | +1.83 | -15.48 |
| Median pair slope vs. OLS4 | +1.61 | -15.66 |

On the 46 complete matches, OLS4 ADE is 8.49 video pixels from camera
histories and 1.60 from publisher histories, against the same targets.
This is a matched input comparison, not an all-trajectory error rate.
The negative lower bounds do not establish negative realized gains;
they show that the available labels cannot establish an improvement.
I did not fit or calibrate a selector on these clips or change the
original 2% rule. The next data task is to improve matched supervision
across several development recordings before fitting a camera risk head.

## Reproduce

[Protocol](protocol.md) | [Full-roster readout](raster_registration/full_online_readout.json)

The commands below use the locally installed publisher data, pinned
detector assets, and frozen Stage 23 split files. The aerial export is
bound to the URL and SHA-256 in `aerial/registration.json`. Place its
ONNX file and ONNX Runtime 1.23.2 (under `runtime/`) in
`data/stage_cvpr2027_experiments/sdd_camera_support_v1/aerial/`.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
py=.venv-pytorch/bin/python
for phase in register rapid mobilenet audit; do
  "$py" scripts/run_m3w_sdd_camera_support.py "$phase"
done
for phase in freeze audit; do
  "$py" scripts/run_m3w_sdd_camera_scale_probe.py "$phase"
  "$py" scripts/run_m3w_sdd_camera_aerial.py "$phase"
done
"$py" scripts/export_m3w_sdd_camera_donors.py
"$py" scripts/audit_m3w_sdd_camera_raster.py
"$py" scripts/audit_m3w_sdd_camera_full_roster.py
```

The first detector pass used native arm64 CPU execution with four
compute threads and no DataLoader workers. The inference commands
resume from per-window files and repeat the first window as a check.
The corrected readouts replay exactly. No reserved WEPDTOF or
DroneCrowd outcome was opened; no deployment or submission changed.
