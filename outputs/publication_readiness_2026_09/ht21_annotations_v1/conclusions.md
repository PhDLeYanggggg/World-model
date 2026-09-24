# HT21/CroHD: dense annotations acquired, forecasting admission withheld

Date: 2026-09-24. Status: **fresh source audit, exact replay verified; no training or prediction evaluation**.

I investigated dense external observations because the completed EqMotion and
simple-motion controls do not establish indispensable neural forecasting or a
useful joint-agent contribution. This step obtains and verifies a real additional
asset. It does not resolve independent risk calibration or improve a model score.

## What actually ran

The publisher's labels-only archive was downloaded, without images, execution of
third-party code or unpacking into the repository. The 53,547,421-byte file has
SHA256 `824e94f1f94af75321103f12d0fc77eb83bde86abb2422c365c3ab0c53161a96`.
All nine sequence metadata files and all four supplied ground-truth files were
parsed. Detector-output files were inventoried but never treated as ground truth.

| Recording | GT rows | Track IDs | Camera motion in INI | Median track rows | Median visible humans/frame |
|---|---:|---:|---|---:|---:|
| HT21-01 | 21,456 | 85 | True | 260 | 48 |
| HT21-02 | 733,622 | 1,276 | True | 413 | 215 |
| HT21-03 | 257,939 | 811 | False | 311 | 249.5 |
| HT21-04 | 175,479 | 580 | True | 227 | 161 |
| Total | 1,188,496 | 2,752 | 3/4 True | Not pooled | Not pooled |

These are fresh counts from the downloaded bytes, not copied benchmark results.
Track IDs are recording-namespaced. There are 62,381 visible static-class rows.
The five supplied test recordings contain no released GT in this archive. Their
forecast evaluation is `not_run`, not zero error. No test detections were used as
pseudo-ground-truth targets.

## History support, not an effective sample size

Complete history uses visible, positive-confidence human observations, merging
classes 1, 2 and 4; class 3 ignore regions are excluded. This is a structural
population, not the benchmark's pedestrian-tracking evaluation population.
Future availability is audited separately and never determines input eligibility.

| History K, stride 1 | Past-eligible queries | Complete future 12 | Partial future 12 | No future 12 |
|---|---:|---:|---:|---:|
| 8 | 1,085,273 | 1,025,577 | 56,026 | 3,670 |
| 16 | 1,044,490 | 990,123 | 50,945 | 3,422 |
| 32 | 973,428 | 926,538 | 43,821 | 3,069 |
| 64 | 857,673 | 820,511 | 34,620 | 2,542 |

With K=8, available raw endpoints +10/+25/+50/+100 number
1,041,043 / 996,155 / 934,009 / 824,065. They are not independent trials.

The cadence sensitivity includes every anchor phase, not one downsampled stream:

| Raw stride | Past span | Future span | Past-eligible | Complete future 12 | Partial | No future |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 7 | 12 | 1,085,273 | 1,025,577 | 56,026 | 3,670 |
| 5 | 35 | 60 | 961,257 | 768,304 | 180,969 | 11,984 |
| 10 | 70 | 120 | 853,957 | 574,567 | 258,994 | 20,396 |

No cadence has been selected for a forecast experiment. These counts cannot be
pooled with SDD's stride-12 main protocol. HT21-03 alone has 220,059 eligible K8
stride-1 queries; it is one recording, not thousands of independent scenes.
Density is present, but it does not prove that joint intervention improves error
or that image-space overlap measures physical collision.

## Why immediate forecasting admission fails

1. **Observation provenance:** the annotations contain interpolated coordinates.
   A past-indexed interface does not make the annotation producer an online
   causal tracker. Keyframe-only eligibility cannot be inferred from a frame
   modulus or from exactly linear coordinate triples.
2. **Population leakage:** the paper defines static people using the complete
   recording. Our input does not carry or exclude that motion class; classes
   1/2 are merged. This removes one known route, not all producer-level concerns.
3. **Geometry:** three released GT recordings declare camera motion. Head-center
   pixels are not ground-plane trajectories. Compensation, scale, homography and
   image-clock correspondence have not been verified. Declared 25 fps is retained
   as metadata, not used to attach seconds or metric claims to M3W results.
4. **Independent sites:** supplied train/test recordings share views or locations.
   Nine recordings do not establish nine independent physical sites. No
   calibration/confirmation allocation has been made.
5. **Use scope:** public download availability is verified; dataset-use terms and
   scientific role admission are not established by that fact alone. No raw
   annotation redistribution is included in Git.

The [source notes](primary_sources.md) separate primary documentation from our
raw-file measurements. The provenance issues are work for this project, not a
request for the user to perform routine audits.

## Implemented checks and result provenance

- `fresh_run`: full acquisition, all-row schema/count audit, cadence analysis,
  28-group independent CSV/set recount, and 96 real-history boundary checks.
- `cached_verified`: complete audit replay, exact cadence replay and exact
  boundary replay. All 1,188,496 rows match the separate recount.
- 36 scoped tests pass, including gaps, future mutation, static-label invariance,
  archive paths and fail-closed data-role requests. The legacy suite was not rerun.
- The intake admission guard refuses training, pretraining, risk calibration,
  official evaluation and confirmation. It is a source-specific guard, not a
  claim that unrelated legacy scripts have been retrofitted.
- `not_run`: HT21 forecasting, learned model fitting, calibration, confirmation,
  metric evaluation, strict online tracking and CrowdTraj acquisition.

All required processes finished. Raw files remain Git-ignored. No deployed model
changed, DroneCrowd remains closed, and neither Stage5C nor SMC ran. The project
still has no independent safety certificate or submission-ready main result.

## Next research action

Resolve source grouping and the observation-producer contract before making
prediction errors visible. A stationary-camera dense diagnostic may be feasible,
but it must be registered as offline head-center forecasting with a fixed cadence
and no physical-safety claim; it cannot replace independent calibration. Continue
seeking accessible, provenance-compatible independent sites rather than treating
overlapping HT21 windows as added statistical independence. CrowdTraj is currently
an acquisition lead with no verified download endpoint, not an available dataset.
