# Can ACME Support a Full Online Scene Comparison?

**9 November 2026.** I sampled 12 disjoint windows from each of two already
exposed recordings, CMU and Miraikan, using only the camera clock. The
earlier diagnostic picked times with reference histories; this one does
not. I kept the frozen [CPU/MPS detector and tracker](../acme_detector_hybrid_v1/README.md)
and the 8-observation / 12-prediction setup at stride four on the
annotation grid. The two schedules share one window per recording; all
16 detector frames in those shared windows agree exactly.

## What Changed

An online history is an identity detected in all eight observed frames.
For reference support, the same online identity must match the same
publisher identity in every observed frame, using the existing
one-to-one ground-anchor criterion. A matched identity is fully scorable
only if the publisher supplies all 12 requested future positions.
Counting two complete identities is different from covering every
eligible identity in a scene. No future positions are scored here.

| Camera and window schedule | Online histories | Stable matches | Complete futures | Windows with a complete pair | Windows with a complete online roster |
|---|---:|---:|---:|---:|---:|
| CMU, camera clock | 400 | 8 | 7 | 1/12 | 0/12 |
| Miraikan, camera clock | 117 | 11 | 8 | 1/12 | 0/12 |
| CMU, previous reference-enriched | 390 | 10 | 10 | 2/12 | 0/12 |
| Miraikan, previous reference-enriched | 108 | 9 | 7 | 1/12 | 0/12 |

All 24 camera-clock windows have at least two online histories. The 517
histories are repeated within two recordings, not 517 independent people.
The previous schedule also lacks full coverage; the two schedules are a
diagnostic contrast, not a matched estimate of a sampling effect.

## Where Support Is Lost

| Camera-clock windows | Complete reference histories before matching | Complete histories with a point match in every past frame, ignoring IDs | Complete histories preserving the online identity |
|---|---:|---:|---:|
| CMU | 51 | 18 | 7 |
| Miraikan | 55 | 30 | 8 |

Each camera has at least two complete *reference* histories in all 12
windows. But 23 of 24 windows have fewer complete reference histories
than the number of eligible online identities. After fixed point
matching, even ignoring online identity, **0/24** windows have enough
complete histories for the full current roster. This is not an
optimized rematching bound. Missing references and detector false
positives can both contribute; unmatched tracks are not all false
positives. For a *pair-only* diagnostic, association still matters:
4/12 CMU and 11/12 Miraikan windows have a complete pair when IDs are
ignored, versus 1/12 each with the current tracker.

**Decision:** Do not score a joint deployment policy on a matched subset
and call it the whole scene. For the full-roster experiment, prioritize
release-matched sources with documented annotation coverage or obtain
exhaustive evaluation labels independently of model scores. Improving
association on these detections can help the pair diagnostic, but cannot
remove the count deficit for the current roster. Neither recording is
added to an independent split. DroneCrowd confirmation stays sealed and
the 2% rule is unchanged. There is no new forecast, risk, metric, or
deployment result.

## Replay

Use the pinned publisher videos, labels and detector from the
[earlier run](../acme_detector_hybrid_v1/README.md). The release
[README](https://huggingface.co/datasets/acmesocnavdataset/ACME-dataset/blob/a0f117c11a1aee25413da95cf8612f5f8209c171/README_ACME.txt)
distinguishes onboard detections from manually verified BEV trajectories.
The latter are evaluation references, not live history inputs.

```sh
.venv-pytorch/bin/python scripts/run_m3w_acme_camera_roster.py register
.venv-pytorch/bin/python scripts/run_m3w_acme_camera_roster.py produce
.venv-pytorch/bin/python scripts/run_m3w_acme_camera_roster.py evaluate
.venv-pytorch/bin/python scripts/audit_m3w_acme_roster_ceiling.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_acme*.py tests/test_m3w_waterloo_intake.py tests/test_m3w_streaming_boxes.py tests/test_m3w_topdown_source_support.py tests/test_m3w_external_causal_sources.py tests/test_m3w_station_and_sharedspace.py
```

Fresh detection: 192 frames in 452.35 seconds on native arm64 with four
CPU threads, zero data workers and the frozen CPU/MPS split. Re-run
`produce` to verify/reuse cached windows and resume any missing ones.
44 focused tests passed. No CREATE job was needed. Raw assets, frame
caches, detector weights and manuscript files are not included.

[Registration](registration.json) | [Roster counts](analysis.json) |
[Reference and association support](support_ceiling.json).
