# Independent Source Qualification: SinD and Foggy Bottom

**2026-11-08.** This is a fresh timing and label-support audit, not a
forecast evaluation. The goal is to find recordings suitable for the
8-observation / 12-prediction protocol without opening DroneCrowd confirmation
outcomes or reusing exposed OpenDD locations as independent evidence.

## What the files support

The script checks byte identity, pedestrian ID, frame/timestamp agreement
and unique track-tick pairs. It selects the earliest nonoverlapping 20-step
intervals with at least two complete 8-step histories. It does not require
any future labels to select an interval or a target. The denominator
includes all eligible targets in those intervals, including those with no
future labels. Candidate windows fit within the release's pedestrian time
extent. Window counts are not independent people, sites, or recordings.

| Source | Released rows | Track IDs | Native queries | Stride-4 queries | Stride-4 targets | Complete 12-step targets | Stride-4 queries with >=2 complete targets |
|---|---:|---:|---:|---:|---:|---:|---:|
| SinD Changchun | 10,451 | 49 | 157 | 43 | 96 | 71 (74.0%) | 28 |
| SinD Chongqing | 15,453 | 40 | 221 | 59 | 147 | 126 (85.7%) | 48 |
| SinD Tianjin | 36,792 | 66 | 564 | 147 | 444 | 413 (93.0%) | 141 |
| SinD Xi'an | 3,419 | 16 | 38 | 10 | 21 | 17 (81.0%) | 8 |
| Foggy Bottom TGSIM | 766,876 | 15,307 | 3,538 | 912 | 5,727 | 2,061 (36.0%) | 540 |

SinD totals: 259 stride-4 queries and 627/708 complete targets (88.6%);
225 queries retain at least two complete targets. Tianjin contributes
444/708 stride-4 targets, so pooled counts should not substitute for
per-city results. Foggy Bottom has 3,251 partial and 415 absent futures.
Its median track has 33 released positions; at stride four a full
8-to-12 interval spans 77 native ticks, including both ends. Even with
many rows, a complete-case-only comparison would discard most targets.
Missing future labels are not proof that a pedestrian left the scene.

The SinD sample clock agrees with `frame_id / 9.99` seconds within
`1e-6` seconds, and the Foggy Bottom clock lies on the 10 Hz grid.
Stride four is therefore approximately 0.4004 seconds and 0.4000 seconds,
respectively, for **these exports only**. This does not establish SDD FPS,
seconds-level SDD horizons, coordinate scale or metric accuracy.

## Why neither is a confirmation set yet

- **SinD:** The four public samples are four city recordings, not the six
  intersections described for [SinD v2](https://arxiv.org/abs/2607.16943v2).
  All four match their pinned Git LFS objects. The publisher's
  [format](https://github.com/SOTIF-AVLab/SinD/blob/930e4dea78d924c6e9a58ff8e378331f93bba8ec/Format.md)
  describes the released ground coordinate fields, but the public
  pedestrian table is `Ped_smoothed_tracks.csv`. The direction and temporal
  support of the position filter are not established by this audit.
  Recomputing velocity from these positions would not certify causal
  inputs. Full data access is [application-based](https://github.com/SOTIF-AVLab/SinD#-dataset-access);
  no request was sent. No terms permitting redistribution of these
  trajectories or derived weights are assumed.
- **Foggy Bottom:** The [USDOT release](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories)
  covers connected intersections in one campaign, not four independent
  validation units. The [field dictionary](https://data.transportation.gov/api/views/brzy-6zfh/columns.json)
  exposes filtered position fields; the [producer report](https://rosap.ntl.bts.gov/view/dot/74647/dot_74647_DS1.pdf)
  describes infrastructure track stitching and interpolation. The audit
  downloaded only pedestrian ID and time; label presence here does not
  measure raw detection coverage or demonstrate causal position histories.
- **TUMDOT-MUC:** The [producer page](https://www.mos.ed.tum.de/vt/forschung/datensaetze/tumdot-muc/)
  describes linked locations and publishes the second acquisition day.
  Archive metadata request returned HTTP 403; bulk contents were not
  audited. No window counts or model results are inferred.

**Decision:** Prioritize checking access to raw frame detections or documented
forward-only position histories for SinD, and confirm capture/source lineage
before assigning any formal role. Foggy Bottom is a lower priority for
independent confirmation given the single linked campaign and missing-label
support. Forecast fitting, transfer, risk calibration and formal confirmation
for these sources are **not_run**. The original 2% rule remains unchanged.
DroneCrowd's 15 calibration / 30 confirmation source-group assignment remains
locked; the prior failed calibration is unchanged and the 30 groups stay sealed.

## Replay

The [input manifest](../../../configs/m3w_external_causal_sources_v1.json)
gives exact bytes and hashes. For each SinD entry use
`https://media.githubusercontent.com/media/SOTIF-AVLab/SinD/<sind_commit>/Data/<member>`
(URL-encode apostrophes) and the listed local filename. Use the Foggy Bottom
query URL verbatim, including the pedestrian class filter. Put the five CSVs
in the ignored local folder `external_data/ExternalTrajectorySourceAudit/`.
Replay fails closed on changes to the snapshots or timing schema.

```bash
.venv-pytorch/bin/python scripts/audit_m3w_external_causal_sources.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_external_causal_sources.py tests/test_m3w_station_and_sharedspace.py tests/test_m3w_topdown_source_support.py tests/test_m3w_external_source_audit.py
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_calibration_roles.py verify
```

Two complete audits matched byte for byte; 30 focused tests passed. The
last command verifies only the locked DroneCrowd role manifest. See
[machine-readable audit](analysis.json) for both strides, track lengths
and missing-label counts. Raw tables are not included.
