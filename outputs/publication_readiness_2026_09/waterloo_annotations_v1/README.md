# Waterloo: how many complete trajectories do two new sites add?

I downloaded the publisher's **database packages only** for the
[Ring Road campus crosswalk](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/datasets/waterloo-multi-agent-traffic-dataset-crosswalk)
and the
[University Avenue / Weber Street intersection](https://uwaterloo.ca/waterloo-intelligent-systems-engineering-lab/datasets/waterloo-multi-agent-traffic-dataset-intersection).
The two downloads total 675,494,980 bytes. I read one database at a time,
checked the schema, and counted only publisher position and timestamp
rows. Types are used for descriptive strata, not model features. No
images, video, velocity exports, future action codes, models or error
scores enter this audit.

| Publisher site | Databases | Position rows, all agents | Pedestrian rows | Pedestrian track IDs |
|---|---:|---:|---:|---:|
| Crosswalk | 17 | 521,133 | 448,483 | 1,153 |
| Intersection | 14 | 2,705,553 | 487,659 | 264 |
| Total | 31 | 3,226,686 | 936,142 | 1,417 |

Track IDs are recording-local, not unique people across recordings.
There are two documented locations, not 31 independent sites. All 31
exports fit a common timestamp grid of 1001/30000 source seconds
with maximum rounding residual below 0.34 microseconds. Every
pedestrian track has consecutive grid ticks between its released
endpoints. This verifies the export clock, not an image-to-clock
reconciliation or a seconds conversion for another dataset.

## Eight Observed, Twelve Predicted

The past alone determines whether a window is counted. The future
counts show how many of those windows have all twelve labels, without
interpolation by our code. The stride-12 diagnostic covers every
anchor phase, not a selected sampling rate for the paper.

| Pedestrian slice | Stride | Past eligible | Complete future | Complete share | Queries with at least two complete pedestrians |
|---|---:|---:|---:|---:|---:|
| Crosswalk | 1 | 440,412 | 426,592 | 96.86% | 78,416 |
| Intersection | 1 | 485,811 | 482,643 | 99.35% | 98,570 |
| Crosswalk | 12 | 352,650 | 194,534 | 55.16% | 44,034 |
| Intersection | 12 | 465,524 | 428,474 | 92.04% | 93,106 |

The large intersection count is not simply more moving pedestrians.
At stride 12, **23.08%** of its complete windows have exactly zero
past path length; the crosswalk has none. The ten longest contributors
account for 11.34% and 2.40% of complete windows, respectively.
A greedy packing of nonoverlapping observation-and-forecast intervals
with two complete pedestrians yields 268 crosswalk and 421 intersection
intervals. These intervals can still share participants and conditions;
neither they nor overlapping windows are independent sites.

## What It Can Establish

Waterloo is a useful candidate for a separate diagnostic on annotation
completeness and stationary-motion mix. It is not admitted for causal
confirmation. Appendix B of
[Sarkar's Waterloo thesis](https://dspacemainprd01.lib.uwaterloo.ca/server/api/core/bitstreams/f845f118-6027-417b-bc3a-02ad8b662ab1/content#page=224)
identifies the release as CC BY-NC 4.0 and attributes geolocated
trajectories to DataFromSky (section B.2, PDF page 226).
The provider describes a
[video post-processing workflow](https://intercom.help/datafromsky/en/articles/3650106-how-does-datafromsky-trafficsurvey-work-faq).
The exact version, temporal smoothing, gap repair and lookahead for
these files are not documented in the inspected sources. Past-indexed
queries alone cannot establish that past coordinates were produced
without looking ahead. UTM coordinates are publisher-described;
geometry and cross-dataset scale have not been independently validated.

The highest-value next source check is whether an unannotated
frame stream or unsmoothed, timestamped detections can be obtained
and matched to the released future labels. The publisher links
annotated videos, not a verified raw observation stream for these
two sites. Among alternatives,
[P-DESTRE](https://p-destre.di.ubi.pt/download.html) documents
frame-level boxes at two campuses, but bundles a large download and
includes attributes not needed for forecasting;
[SCOUT](https://scout.epfl.ch/download) offers camera calibration
and tracking labels but requires download authorization and does not
add multiple documented physical locations. Neither was acquired.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/audit_m3w_waterloo_intake.py
.venv-pytorch/bin/python scripts/audit_m3w_waterloo_intake.py --replay
.venv-pytorch/bin/python scripts/verify_m3w_waterloo_motion_support.py
```

[Machine-readable audit](analysis.json) includes official URLs,
archive/member hashes and all per-recording counts. [Separate recount](motion_support.json)
recomputes each pedestrian count from continuous track lengths and
checks simultaneous support; it is a second implementation, not an
external replication. The full audit and recount replayed exactly;
19 scoped tests passed. Raw data stay local. No training, scientific
roles, risk recalibration or independent model evaluation ran.
The original 2% rule and sealed 15/30 DroneCrowd assignment are unchanged.
