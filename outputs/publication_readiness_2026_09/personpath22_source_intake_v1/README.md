# PersonPath22: Source Overlap and a Camera Coverage Check

**9 November 2026.** I checked the public
[PersonPath22 release](https://registry.opendata.aws/person-path-22/) and
its [publisher downloader](https://github.com/amazon-science/tracking-dataset/blob/2f9c757f847e172ce375372913bdbb3a47074e4f/download.py).
The original dataset is described by Shuai et al.,
[Large Scale Real-World Multi-Person Tracking, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136680493.pdf).
I opened 138 per-clip training annotation tables. For the 98 official
test videos I used only the public split, source mapping and aggregate
metadata; no test annotation member was opened.

## Split and Timing

- The public 12.24 GB camera archive directly contains **107 training
  and 74 test videos**; the rest have external source mappings.
  A verified byte-range read of the catalog used 659,832 bytes.
- Three train/test pairs are crops of the **same original MEVA video**.
  Four documented MEVA camera codes and one VIRAT camera code appear on
  both sides. The original crops need not overlap in frames for this to
  be a problem for *new-scene* evaluation. The other clip IDs do not
  establish physical independence by themselves.
- The [release code](https://github.com/amazon-science/tracking-dataset/blob/2f9c757f847e172ce375372913bdbb3a47074e4f/README.md)
  documents a 5 Hz public keyframe release. Native frame IDs and
  recorded millisecond fields agree within 1.05 ms in all 138 training
  clips. **68 clips** also meet the stricter check that all rows fall
  on a common 5 Hz grid within half a native frame plus 1 ms. The other
  70 contain off-grid labels. I did not snap or interpolate them.

The following is annotation-only support, not camera or model coverage.
The camera extent fixes nonoverlapping 8-observation / 12-prediction
intervals on every other 5 Hz tick. Each identity present in all eight
observed ticks stays in the denominator whether or not the future exists.

| Training clips meeting the strict clock check | Windows | Complete reference pairs | Complete reference rosters with at least two agents |
|---|---:|---:|---:|
| All 68 clock-compatible clips | 222 | 212 | 58 |
| 56 directly available in the camera archive | 157 | 156 | 25 |

The 25 whole-reference windows occur in 13 clips, not 25 independent
sites. None of the 56 has a physical camera code in the supplied
external mapping. This does not prove they share a camera; it means
their site independence needs evidence beyond the UID.

## Real Camera Input

I used the first numeric UID among clock-qualified, directly available
training clips with a complete camera interval, not the best score or
most futures. PyAV decoded `uid_vid_00002.mp4` at **24 FPS, 604 frames**;
the presentation timestamps agree with its release clock. The frozen
COCO ResNet50-FPN detector processed only 24 past frames, with causal
transport and no annotation input. All three intervals have multiple
online identities. On the exact same detector and association outputs,
I compared the publisher's amodal and visible box releases at IoU 0.5.

| Evaluation reference | One-frame matched pairs | Complete reference histories | Complete matches ignoring online IDs | Complete stable online matches | Whole online roster |
|---|---:|---:|---:|---:|---:|
| Amodal | 201 | 39 | 2 | 1 | 0/3 |
| Visible | 263 | 6 | 1 | 1 | 0/3 |

There are 44 online histories; neither box definition gives a stable
complete pair. The two releases cover different reference populations.
More visible-box matches do not repair future support, and association
alone cannot recover more than the fixed framewise matches in this
diagnostic. No forecast or risk result is reported for this clip.

## FE-DETRAC Availability

The [FE-DETRAC portal](https://scidm.nchc.org.tw/dataset/fe-detrac)
describes the tracking resource in Huang et al.,
[FE-DETRAC, IEEE Internet of Things Journal, 2026](https://doi.org/10.1109/JIOT.2026.3711126).
The two linked public ZIP catalogs I checked each contain 8,000 images
and per-image labels, not an identifiable MOT index. In the newer
ZIP's COCO training table I found 5,288 images and no track-ID or
video/timestamp field. This is evidence about the linked archives,
not a claim that the 20,000-frame tracking release does not exist.
The full tracking resource and its physical-site map are still needed.

**Decision:** Do not use the published PersonPath22 split as independent
scene evidence. Prioritize a source with verified physical-group
membership and complete camera references before committing to another
cross-scene model run. Keep the existing 2% rule, the original source
reservations and the current paper unchanged.

## Replay

```sh
.venv-pytorch/bin/python scripts/audit_m3w_personpath22_sources.py --include-fisheye-catalogs
.venv-pytorch/bin/python scripts/run_m3w_personpath22_camera_probe.py prepare
.venv-pytorch/bin/python scripts/run_m3w_personpath22_camera_probe.py produce
.venv-pytorch/bin/python scripts/run_m3w_personpath22_camera_probe.py evaluate
.venv-pytorch/bin/python scripts/compare_m3w_personpath22_box_references.py
```

The pilot reuses the registered COCO detector and the existing local
PyAV 18.1.0 runtime at `data/stage_cvpr2027_experiments/media_decode_runtime`.
It runs on arm64 CPU/MPS with four Torch threads and no data workers.
The three inference checkpoints took 62.2 seconds. The initial attempt
stopped before any inference checkpoint because of a stale seek call;
registration v2 reused the verified video and completed the run.
Source support and box comparison replayed unchanged. Raw assets and
checkpoints remain local; no CREATE job was necessary.

[Source audit](analysis.json) | [Camera registration](camera_registration_v2.json) |
[Coverage](camera_probe.json) | [Box comparison](camera_box_reference_comparison.json).
