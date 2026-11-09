# MOT20 Camera Pilot: Better Association, Still No Full Roster

**9 November 2026.** I tested the longer public indoor recording,
MOT20-02, from the [publisher's archive](https://motchallenge.net/data/MOT20/).
The publisher lists 25 FPS. The sequence metadata and the complete
numbered image inventory agree with the published 2,782-frame length.
I used 13 disjoint 8-observation / 12-prediction intervals at stride
10, fixed from the camera clock. No interval was selected from labels
or model errors. The 0.4-second sampling interval applies to this
recording, not to SDD raw-frame horizons.

## Association on the Same Detector Boxes

I ran the existing frozen COCO ResNet50-FPN detector on only the eight
past frames per interval. I did not use the publisher's detections:
the [MOT20 paper](https://arxiv.org/abs/2003.09003) says their detector
was trained on the benchmark's training set, which includes this clip.
I compared last-box association with two-point causal transport on
the same camera detections. Only after the caches were complete did I
load the release's reference boxes.

The matcher converts the publisher's one-based image coordinates to
the detector's zero-based raster, then uses one-to-one box IoU at
0.5. An eligible history has an online ID in all eight observed
frames; a stable match has the same active pedestrian reference ID
at all eight frames. A complete match also has all 12 future reference
positions. No reference position or ignore class enters tracking.

| Association | Online histories | Stable matches | Complete futures | Windows with a complete pair | Whole online roster |
|---|---:|---:|---:|---:|---:|
| Last box | 668 | 35 | 26 | 6/13 | 0/13 |
| Causal transport | 727 | 47 | 35 | 10/13 | 0/13 |

There are 540 complete released reference histories across the 13
windows; 118 of them match a point-in-time box in every observed frame
under the fixed assignments when online identity is ignored. This
includes a complete pair in every window. Association therefore matters
for recovering pairs. It does not solve full coverage: in 12/13 windows
the number of complete references is already below the transport roster
size, and in 13/13 even ignoring identity does not recover enough
complete matches. Excluding identities that consistently match a
[TrackEval distractor class](https://github.com/JonathonLuiten/TrackEval/blob/master/trackeval/datasets/mot_challenge_2d_box.py)
in all eight past frames removes eight histories from the transport
denominator, but leaves the whole-roster result at zero. This is a
label-based diagnostic, not an inference rule or the official MOT score.

## Paired Motion Baselines

On the **same 35 complete transport histories** in 12 windows, I
forecast from the bottom centre of the detector boxes. Each window
receives equal weight.

| Method | ADE (image px) | FDE (image px) |
|---|---:|---:|
| Last-step constant velocity | 119.64 | 206.56 |
| Anchored four-step least squares | 100.87 | 169.78 |

OLS4 has lower mean FDE in 11 of the 12 scored windows. An oracle
choosing between the two by each target's eventual FDE reaches 158.42
px; this is a hindsight bound, not a learned policy. These are complete-case
results from one crowded indoor recording, not 12 independent scenes.
The independent joint-policy test, original 2% risk estimate and deployment
remain unchanged.

**Next:** Keep causal transport and OLS4 as candidates for further
camera development. For a new independent scene comparison, prioritize
complete annotation of the online roster and enough distinct sites.
The public MOT20 training clips are not four independent sites: the
source paper describes three sites in the full benchmark, with one
reserved for the unpublished-label test split.

## Replay

```sh
.venv-pytorch/bin/python scripts/run_m3w_mot20_camera_support.py prepare
.venv-pytorch/bin/python scripts/run_m3w_mot20_camera_support.py produce
.venv-pytorch/bin/python scripts/evaluate_m3w_mot20_camera_support.py
.venv-pytorch/bin/python scripts/replay_m3w_mot20_motion_diagnostic.py
```

Use the separate evaluator above for the one-based coordinate conversion.
The motion replay converts a NumPy count to a JSON integer without
changing the frozen computation. Both readouts replayed exactly on the
same caches. Range retrieval transferred 144,659,240 bytes for the
registered camera images and metadata, rather than the 5 GB archive.
Native arm64 CPU/MPS inference completed 104 frames in 245.24 seconds
with four CPU threads and zero workers. Re-run `produce` to resume the
matching backend from per-window checkpoints. 51 focused tests passed.
No CREATE jobs, raw images, label files, or weights are included.

[Camera registration](registration.json) |
[Evaluation registration](evaluation_registration.json) |
[Coverage](analysis.json) | [Motion scores](motion_diagnostic.json).
