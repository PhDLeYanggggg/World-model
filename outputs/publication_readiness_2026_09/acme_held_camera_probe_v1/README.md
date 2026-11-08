# Frozen Camera Transfer: CMU and Miraikan

**2026-11-08.** Transfer of the frozen GMU detector and association pipeline
to two new cameras. This is a diagnostic of observable histories, not a
confirmation of the M3W decision policy.

## Locked Setup

From the [pinned public ACME release](https://huggingface.co/datasets/acmesocnavdataset/ACME-dataset/tree/a0f117c11a1aee25413da95cf8612f5f8209c171),
I chose the smallest video with a matching pixel-trajectory table at each
of CMU and Miraikan. Selection did not use predictor scores. Neither
camera contributed to the GMU association changes. There are two camera
units, not 24 independent cameras. GMU and Michigan are kept as prior
development comparisons and not pooled with these results.

For each camera I registered 12 disjoint windows before detection,
using the fixed GMU sampling rule: quantiles of starts with two complete
past histories in the release, regardless of future availability. The
schedule is enriched for reference presence; it is not an estimate of
coverage over all camera frames. Each window contains 8 observed and
12 requested steps, every fourth 10 Hz annotation tick. Camera frames
are mapped from the first frame at 1:1 for CMU and 6:1 for Miraikan.
Both released video and reference mask rasters are 1920 by 1080.
Presentation timestamps are checked at every sampled frame.

All detections and association use images alone, with the frozen COCO
detector and the original last-box and two-point-transport trackers.
The reference match uses the same bottom-centre anchor and distance
criterion as at GMU. A stable history requires the same online and
reference identity across all eight observations. The unchanged
engineering floor is at least eight stable histories in three windows
**within each camera**. Future labels are scored only on a camera that
clears that floor. References are publisher-edited trajectories, not
live inputs. Baseline errors, where available, are in image pixels.

## Results

All 192 sampled frame timestamps matched the registered grid to
within 0.1 ms. The original and transported association arms used the
same detector boxes.

| Camera | Reference 8-step histories | All 8 points matched, ignoring online IDs | Stable original / transported | Windows with stable transport | Complete futures |
|---|---:|---:|---:|---:|---:|
| CMU | 60 | 17 | 8 / 8 | 5 / 12 | 7, in 4 windows |
| Miraikan | 76 | 7 | 1 / 1 | 1 / 12 | Not scored |

The extra stable history obtained on the GMU development clip did **not**
repeat as a count gain on either camera. CMU clears the engineering floor;
Miraikan does not. Even ignoring track IDs, most released histories lack
a reference match in at least one of the eight frames. Association alone
cannot close that gap under the frozen criterion. An unmatched detection
is not necessarily a false positive because the release is not an
exhaustive annotation of the detector's population.

| CMU, same 7 complete targets, equal weight across 4 windows | ADE (px) | FDE (px) |
|---|---:|---:|
| CV | 37.28 | 70.49 |
| Anchored OLS4 | 34.82 | 64.75 |

OLS4 reduces ADE by 6.6% and FDE by 8.1% on this conditional subset.
One of the eight stable CMU histories has an incomplete future.
These are **not** camera-population error estimates. I report no
cross-camera confidence interval, risk pass, or neural-model gain.
The independent M3W evidence gap is unchanged.

**Decision:** Keep the simple motion comparison for planning; do not
promote GMU transport as a general camera-input repair. Next prioritize
a candidate with better verified online observation coverage or
publisher per-frame detection streams. It is not useful to train a
forecaster on these few complete cases or to weaken the support or
2% deployment rule to make the experiment scoreable.

## Replay

The [registration](registration.json) binds the source filenames,
SHA-256s, clocks, detector weights, unchanged GMU code and start ticks.
Use the publisher \`resolve\` endpoint at the pinned revision and save
each video and TOML under its registered local stem in ignored
\`external_data/ACME_source_audit/\`. Raw videos, raster masks,
annotations, weights and checkpoints are not in Git.

\`\`\`sh
.venv-pytorch/bin/python scripts/run_m3w_acme_held_camera_probe.py register
.venv-pytorch/bin/python scripts/run_m3w_acme_held_camera_probe.py produce --resume
.venv-pytorch/bin/python scripts/run_m3w_acme_held_camera_probe.py evaluate
\`\`\`

Production uses native arm64 Torch, four CPU threads and no worker
processes. It checkpoints each 8-step prefix and writes a heartbeat.
No fitting, threshold search, novel metric scale, policy promotion,
Stage5C or SMC. DroneCrowd's 30 groups remain sealed and the original
2% rule is unchanged.
The analysis replay matched exactly on the frozen sensor caches.
51 focused tests passed, including frame mapping and equal-query score
weighting with missing futures.
[Machine-readable results](analysis.json).
