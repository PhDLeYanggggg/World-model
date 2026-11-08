# ACME Detector Repair on CMU and Miraikan

**8 November 2026.** A stronger frozen detector repairs the observation
coverage check on two already exposed ACME cameras. This is a camera
input diagnostic, not independent evaluation of the forecast policy.

## Fixed Comparison

Replace the original MobileNet-FPN detector with COCO-pretrained
Faster R-CNN ResNet50-FPN v2. Keep the exact 12 disjoint, reference-enriched
windows per camera, 8 observed / 12 requested steps, decoded frames,
outer thresholds, tiling, track association and reference match.
All 192 frame identifiers, pixel-array hashes and presentation timestamp
errors matched the original run. Pixels and past-only online association
produce the histories; the publisher-edited trajectories are used only
after the caches are complete, to check matches and score futures.
[Original comparison](../acme_held_camera_probe_v1/README.md).

| Camera | Reference histories | All 8 points matched, ignoring online IDs (old → new) | Stable identities (old → new) | Windows with stable identity (old → new) | New complete futures |
| --- | ---: | ---: | ---: | ---: | ---: |
| CMU | 60 | 17 → 18 | 8 → 10 | 5/12 → 8/12 | 10 in 8 windows |
| Miraikan | 76 | 7 → 38 | 1 → 9 | 1/12 → 6/12 | 7 in 5 windows |

Both cameras now clear the unchanged engineering floor of eight
stable histories in at least three windows. The difference between
point matches and stable identities remains large at Miraikan. The
release is not an exhaustive record of detections, so unmatched
online tracks are not necessarily false positives.

| Camera | CV ADE / FDE (px) | Anchored OLS4 ADE / FDE (px) |
| --- | ---: | ---: |
| CMU | 20.74 / 30.87 | 15.11 / 16.38 |
| Miraikan | 27.46 / 56.69 | 27.88 / 49.32 |

Within each camera, the baselines are paired on exactly the same complete
futures; windows receive equal weight. OLS4 wins both errors on CMU,
but trades slightly worse ADE for better FDE on Miraikan. Do not compare
these errors across detectors as paired effects: changing the detector
also changes the scored population. There are only two exposed camera
units and 17 complete histories, not 24 independent units. No
population estimate or cross-camera confidence interval is supported.

## Runtime and Replay

- `fresh_run`: 24 windows, 192 frames, 464.45 seconds on native arm64.
  Four CPU threads and zero data workers; MPS runs the backbone/FPN and
  ROI box head. Proposal selection, matching and tracking stay on CPU.
  An all-MPS pilot did not finish: sampling located a wait in GPU NMS
  sorting. It was stopped and supplied no results. The first comparable
  window took 19.248 s versus 103.391 s on CPU alone (5.37× for that
  window; there is no full CPU speed comparison).
- `cached_verified`: the five independently produced CPU windows have
  matching frame hashes, counts and ordered track identities across all
  40 frames, identical support, maximum box-coordinate difference
  0.001221 px and maximum score difference 0.00000906. CPU and MPS
  checkpoints were not mixed. Resume reused all 24 hybrid windows,
  with zero new detections, and evaluation reproduced identical JSON.
- `not_run`: full CPU readout, neural forecast evaluation, deployment
  risk evaluation and independent confirmation. No model was trained.

```sh
.venv-pytorch/bin/python scripts/run_m3w_acme_hybrid_detector.py produce
.venv-pytorch/bin/python scripts/run_m3w_acme_hybrid_detector.py evaluate
```

Use the videos and detector weights pinned by the
[CPU registration](../acme_detector_repair_v1/registration.json) and the
original MobileNet prefix caches. `produce` automatically verifies and
reuses finished windows; it writes an atomic heartbeat under the ignored
`data/stage_cvpr2027_experiments/acme_detector_hybrid_v1/`. Raw videos,
annotations, frame-level caches and weights are not in Git.
67 focused regression tests passed across both checks this session.
[Hybrid registration](registration.json) | [Scores and support](analysis.json).

**Decision:** Keep the detector repair as engineering evidence. Next
verify that an annotation-independent, source-disjoint camera roster
can sustain enough stable identities and complete targets for the
registered evaluation. Do not lower the gate or treat these exposed
windows as confirmation. The original 2% rule, DroneCrowd confirmation
seal and deployment are unchanged. Errors here are image pixels; no
metric, seconds-level, true 3D or foundation claim. No Stage5C or SMC.
