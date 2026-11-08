# ACME Camera Support: Diagnosis and Repair Trial

**2026-11-08.** One exposed Michigan camera, two nonoverlapping 32-sample
segments at stride four (raw starts 0 and 1600). This is a camera-support
experiment, not a transfer or forecast-accuracy experiment. The later segment
is from the same camera, not a new site.

## Cause of the Original Mismatch

I overlaid publisher points and frozen detector boxes on six of the first
segment's frames. The video shows an **oblique corridor view** rather than a
nadir aerial view. The detector repeatedly fired on a portrait on the left
wall and missed visible blurred pedestrians farther down the corridor. The
points were near the ground rather than near box centres. This is concrete
evidence of false support and missed detections; it does not prove a precise
camera-to-label timing calibration or prove every unmatched point is a miss.

The [producer paper, Sec. 4.2](https://arxiv.org/html/2607.21964v1#S4.SS2)
describes privacy-related visual changes and the upstream YOLO/ByteTrack
outputs. I did not find a per-frame pre-anonymization BEV detection stream
for this camera in the [pinned publisher release](https://huggingface.co/datasets/acmesocnavdataset/ACME-dataset/tree/a0f117c11a1aee25413da95cf8612f5f8209c171/BEV).
This does not establish that no such stream exists elsewhere.

## Repair Readout

I changed the scoring anchor to the bottom centre of the box and required
one-to-one associations within `max(12 px, 0.22 * box height)`. I compared
the original detector, a filter using the producer's static walkable polygons,
and the same filter with a second inference pass on two smaller image crops.
The foreground probe uses chronological video only and does not use labels
in detection, background updates or tracking. Its association criterion and
windows are the same. Counts include all released points in these windows;
the evaluated population is not silently trimmed to the floor filter.

| Input | Detections early / later | Matched points early / later | Online 8-step histories early / later | 8-step histories matched to one identity |
|---|---:|---:|---:|---:|
| Original detector | 69 / 45 | 4 / 1 | 26 / 25 | 0 / 0 |
| Ground region only | 37 / 13 | 4 / 1 | 1 / 0 | 0 / 0 |
| Ground region + distant crops | 39 / 17 | 4 / 1 | 2 / 0 | 0 / 0 |
| Past-only foreground + ground region | 65 / 20 | 7 / 7 | 1 / 0 | 0 / 0 |
| Released reference support | 68 / 23 points | — | 47 / 3 label histories | — |

The floor region removed 32 detections in each segment without reducing
matched points. Most original 8-step histories disappeared: the earlier
camera prefix replay demonstrated computational reproducibility, **not**
reliable observed people. The distant crop did not add matches. Foreground
added some point support but fell short of even eight total matches in
either segment. Under this fixed matching rule, changing the tracker alone
cannot construct a verified eight-step history from any tested detection set.
Foreground also cannot be assumed to cover stationary pedestrians.

**Decision:** Do not use this camera/detector combination for forecast
validation. Prefer publisher pre-anonymization online detections with
documented timestamps and a validated ground anchor, or a better exposed
camera with an independently specified coverage check before forecasting.
Do not repair the deficit by feeding offline corrected tracks as live
inputs or by scoring only the few matched points. This finding does not
disqualify all ACME sites.

## Replay

Use the video and annotation snapshots in the [prior intake](../acme_prefix_intake_v1/README.md).
Save the publisher's [static polygons](https://huggingface.co/datasets/acmesocnavdataset/ACME-dataset/blob/a0f117c11a1aee25413da95cf8612f5f8209c171/BEV/BEV_annotations/annotations/umich_GX010036_BEV1_DOW_transformed.json)
locally as `external_data/ACME_source_audit/umich_floor_regions.json`.
The script checks the publisher snapshots and records its run configuration.

```sh
.venv-pytorch/bin/python scripts/run_m3w_acme_camera_repair.py
.venv-pytorch/bin/python scripts/run_m3w_acme_foreground_probe.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_acme_camera_repair.py tests/test_m3w_acme_prefix_sources.py tests/test_m3w_external_causal_sources.py tests/test_m3w_station_and_sharedspace.py tests/test_m3w_topdown_source_support.py tests/test_m3w_external_source_audit.py tests/test_m3w_streaming_boxes.py tests/test_m3w_waterloo_intake.py
```

The foreground replay matched exactly and 44 focused tests passed.
[Detector comparisons](analysis.json) | [Foreground](foreground_probe.json) |
[Detector registration](registration.json).
No frames, raw trajectories or learned weights are included.
Forecast accuracy, deployment risk and independent confirmation are **not_run**.
The 2% rule and DroneCrowd's 30 sealed groups are unchanged. Stage5C and SMC
remain off.
