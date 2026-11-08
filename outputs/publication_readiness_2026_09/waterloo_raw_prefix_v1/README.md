# Can I build causal inputs from Waterloo's original video?

Yes, for a bounded engineering sample. I processed the unannotated
`DJI_0669.MP4` from the [Waterloo roundabout sample release](https://wiselab.uwaterloo.ca/multi-agent-roundabout-traffic/raw_videos/Sample%20Dataset/)
without opening its same-named tracking log for labels. This site is
different from the crosswalk and intersection in the earlier audit.
I do not treat detection IDs as ground-truth identities.

## Raw video to forecast

From each sampled frame I ran a fixed tiled version of
[Torchvision's COCO MobileNetV3 Faster R-CNN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn.html)
on local arm64 CPU. I used current detections and the most recent
available matched boxes to associate IDs, with no retrospective
merges or gap filling. Eight consecutive sampled detections are
required before emitting CV and OLS4 paths for the next twelve
sampled steps. The centers are in image pixels. No learned M3W
policy, detector training, coordinate calibration, or label scoring ran.

| Engineering check | Fresh result |
|---|---:|
| Sampled frames (raw indices 0 through 252, stride 4) | 64 |
| Detection rows | 663 |
| Assigned track IDs | 114 |
| History-complete forecasts (each emits two baseline paths) | 289 |
| Forecasts of detector-class person | 57 |
| Exact cold replay | 24 frames, 99 forecasts |

The second process stops decoding after the first 24 sampled frames.
Their decoded-pixel, detection, track assignment and forecast digests
match the longer run exactly. This is evidence of prefix invariance
of the implemented frame-level pipeline, not a claim about decoder
latency or tracking accuracy. The run took 85 seconds, including
asset hashing; the cold replay took 32 seconds. The fixed pilot
stride is not a benchmark cadence decision.

## Candidate target pair

The [publisher's raw directory](https://wiselab.uwaterloo.ca/multi-agent-roundabout-traffic/raw_videos/2021.2.5%20recording/)
contains `DJI_0932.MP4`; its [SQLite package](https://wiselab.uwaterloo.ca/multi-agent-roundabout-traffic/dataset/)
contains `932.db` with internal file ID 932. I downloaded both
and verified that the video has 1,220 frames and that the
31,962 annotation timestamps fit its nominal 24000/1001 clock.
The database includes six pedestrian tracks. It does not include
per-frame pixel boxes or a verified transform from its trajectory
coordinates to our detector centers. Filename and clock agreement
do not identify the frame offset, geometry or per-agent match.

The sample log `DJI_0669.tlgx` is not the `932.db` recording and
`669.db` is absent from the 26-member package. The
[vendor export documentation](https://intercom.help/datafromsky/en/articles/3457473-how-to-export-raw-telemetry-trajectory-data-export-trajectories-to-csv)
describes an image-coordinate export from its desktop viewer. The
shortest remaining path is to obtain that export for a verified
raw/log pair, then check frame offset and target point definition on
past-only correspondences before evaluating future labels. The
vendor viewer was not run and no sample label export was produced.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_waterloo_raw_prefix.py build
.venv-pytorch/bin/python scripts/run_m3w_waterloo_raw_prefix.py verify
.venv-pytorch/bin/python scripts/audit_m3w_waterloo_raw_pair.py
```

Exact asset URLs and hashes are in [registration](registration.json)
and [pair preflight](paired_source_preflight.json). An interrupted
build can resume with `build --resume` and hash-verified frame
checkpoints. The [run](analysis.json) and [cold verification](verification.json)
contain only counts and digests. Five targeted tests passed; the
pair preflight also replayed exactly. Raw clips, annotations,
detector weights and per-agent forecasts remain local.

The experiment does not change the approved data roles, original
2% rule, or sealed DroneCrowd confirmation. No new deployment,
metric result, Stage5C or SMC.
