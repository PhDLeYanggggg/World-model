# ACME: A New Route to Independent 8-to-12 Evaluation

**2026-11-08.** Fresh release and camera-input audit. There is no forecast
accuracy, risk, adaptation or independent confirmation result here.

## Public Release and Timing

[ACME](https://huggingface.co/datasets/acmesocnavdataset/ACME-dataset) now has a
public, ungated CC BY 4.0 dataset card. I pinned revision
`a0f117c11a1aee25413da95cf8612f5f8209c171`. The file listing contains 127
pre-transform pixel trajectory TOMLs across seven collection sites, totaling
875,147,625 bytes, plus a camera-intrinsics sidecar. This supersedes the
[September access check](../acme_source_candidate.md) for this revision.

I acquired the lexicographically first TOML from each site and verified the
seven Git blob or LFS hashes against the pinned publisher listing. The
files contain 1,237,914 positions and 6,393 clip-local track records. I used
the explicit frame list only when it was strictly increasing, matched the
position count and agreed with the stated first frame. I did not repair
frames or use demographics. 5,817 tracks passed those checks; 351 had
different coordinate and frame-list lengths, 207 lacked an explicit clock,
and 18 had no positions. On the consistent subset, frame numbering and row
ordinal agree. The concern is the ambiguous subset, not a proven frame
shift in all ACME tracks.

| Site sample | Consistent / total tracks | Nonoverlapping queries | History-eligible targets | Complete 12-step futures |
|---|---:|---:|---:|---:|
| Bonn | 441 / 488 | 190 | 911 | 664 |
| CMU | 223 / 330 | 153 | 367 | 247 |
| GMU | 122 / 156 | 68 | 157 | 102 |
| Keio | 1,048 / 1,109 | 242 | 4,088 | 3,533 |
| NUS | 1,038 / 1,170 | 64 | 2,037 | 1,514 |
| Miraikan | 2,859 / 3,054 | 458 | 4,477 | 3,015 |
| Michigan | 86 / 86 | 27 | 54 | 31 |
| **Total** | **5,817 / 6,393** | **1,202** | **12,091** | **9,106 (75.3%)** |

Stride four on the publisher's nominal 10 Hz annotation clock. Within each
audited file, the earliest nonoverlapping 20-sample interval is selected
when at least two tracks have complete 8-step histories; future presence
does not select the interval or target. The 2,682 partial and 303 absent
futures stay in the support denominator. Selecting the consistent subset
is a retrospective **file-quality audit**, not an online track filter.
Counting by row ordinal instead would give 1,256 queries and 10,391
complete futures, but those counts include unvalidated timing. Windows and
track IDs are not independent sites or people. Seven samples do not
establish the support of all 127 files.

## Input Provenance

The [ACME project](https://raoshashank.github.io/acme-socnav-dataset/) and
[pinned reader](https://github.com/social-navigation-group/sbpd-dataset-pipeline/blob/8ec39f0adce6bc24ea8ac805915b2b966cd93fcf/acme_dataset/bev.py)
distinguish the editable pixel annotations from transformed trajectories.
The latter can use short-track filtering, interpolation and smoothing;
the former are **before that transform**, but were already manually
reviewed, including joins and breaks. Do not pass either off as live
detector histories. Match online detections to labels for scoring only,
without using future associations for input construction or eligibility.

For SinD, the [original producer paper, Sec. III.D](https://arxiv.org/abs/2209.02297)
explicitly describes Rauch-Tung-Striebel smoothing, including a pedestrian
motion model. The [v2 description](https://arxiv.org/html/2607.16943v2#S3.SS2)
mentions recursive filtering but does not establish a forward-only
preprocessing chain for each pinned CSV. The public source tree at the
[sample commit](https://github.com/SOTIF-AVLab/SinD/tree/930e4dea78d924c6e9a58ff8e378331f93bba8ec/Data)
has smoothed tracks and static maps but no matching raw video or
pre-smoothing pedestrian positions. Even the script named
[`rebuild_cc_pedestrians_from_raw.py`](https://github.com/SOTIF-AVLab/SinD/blob/930e4dea78d924c6e9a58ff8e378331f93bba8ec/scripts/rebuild_cc_pedestrians_from_raw.py)
reads the smoothed CSVs. Changing the velocity formula cannot undo this.
I am prioritizing ACME's camera route over using SinD's processed positions.

## Camera Prefix and Label Alignment

I downloaded one 489.7 MB matching Michigan BEV video and checked the
publisher SHA-256. On native arm64 CPU, a frozen COCO person detector and
current-frame tracker processed the first 32 samples at stride four. The
container reports 10 FPS, and decoded timestamps for the sampled prefix
advance by 0.4 seconds. The run yielded **69 person detections and 26
8-to-12 forecast instances** without opening annotations. Starting from
scratch on only the first 16 samples reproduced all 35 detections and 10
forecasts there exactly. These are engineering counts, not accuracy.

As a separate geometry-only check, I compared 68 released annotation
points from those same 32 frames to the 69 cached boxes. Same frame index
and unchanged pixel coordinates gave only **3 one-to-one point-in-box
pairs**; the median nearest-center distance was 181.9 pixels. The check
does not identify a unique cause: camera/annotation registration, track
coverage and detector misses are all possible. It does establish that
direct identity alignment is not verified, so I did not score the
forecasts. No time offset, transform or score threshold was fitted.

Next, verify the source camera, frame origin and any image transform on
this engineering clip. Verify detection-to-label association separately
before freezing new sites for a forecast comparison against CV and OLS4.
No new formal calibration or confirmation roles are assigned. The
original 2% rule and DroneCrowd's 30 sealed groups are unchanged. No
seconds or metric claims carry over to SDD.

## Replay

The tree request is the publisher API at the pinned revision:
`/api/datasets/acmesocnavdataset/ACME-dataset/tree/<revision>/BEV/BEV_annotations/trajectory-transform/trajectories?recursive=true&expand=false`.
Save its response as `trajectory_tree.json` in ignored local
`external_data/ACME_source_audit/`. Download the first TOML per site shown
in [the machine audit](analysis.json) from the pinned `resolve` paths as
`<site>.toml`. The video URL and hash are in [the probe registration](video_probe_registration.json);
the frozen COCO detector is the same checkpoint as in the Waterloo
camera probe. Raw annotations, video, images and weights stay local.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_acme_prefix_sources.py
.venv-pytorch/bin/python scripts/run_m3w_acme_video_prefix.py build
.venv-pytorch/bin/python scripts/run_m3w_acme_video_prefix.py verify
.venv-pytorch/bin/python scripts/audit_m3w_acme_video_alignment.py
```

The build checkpoints detections and writes a heartbeat. Pass `--resume`
to continue with the same source and model. Verification re-runs the
detector without the cache. No CREATE resources are needed for this probe.
The seven-file count replay was byte-identical and 42 focused tests passed.
[Camera results](video_probe.json) | [Cold replay](video_probe_verification.json) |
[Pixel alignment check](video_probe_alignment.json).
