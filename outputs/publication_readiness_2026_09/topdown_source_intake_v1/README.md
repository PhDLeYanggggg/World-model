# Independent Top-Down Source Support

**Status: `fresh_run`, structural intake only, exact replay matched for both sources.**
No predictor was fitted or scored; no new data roles were assigned. DroneCrowd
remains at 15 calibration groups and 30 sealed confirmation groups. The
original 2% rule is unchanged. No deployment change, Stage5C execution or SMC.

## What Was Counted

A query can enter the count if its complete 20-sample grid lies within the
recorded timeline. Each target must have all eight observed samples; none of
its future samples is required to enter. I counted endpoint availability and
complete, partial or absent 12-step labels separately. A multi-agent query
requires at least two tracks from the same class. For a non-overlapping
multi-agent count, the earliest qualifying complete query is taken and its
entire grid interval is skipped before taking the next. Neither rolling nor
non-overlapping queries are counted as independent acquisition sites.

Only released annotation IDs, class, timing and file structure were inspected.
The OpenDD clock check covers pedestrian and bicycle timestamps and recording
extents, not every timestamp from other classes. Position columns were not
parsed; no images, forecast errors or confirmation labels were opened.

## AerialMPT: Dense Labels, Short Clips

The [DLR release](https://www.dlr.de/en/eoc/about-us/remote-sensing-technology-institute/photogrammetry-and-image-analysis/public-datasets/aerialmpt-a-dataset-for-pedestrian-tracking-in-aerial-imagery)
contains 14 clips, 307 image frames, 44,740 annotation rows and 2,528 clip-local
track IDs. The [source paper](https://elib.dlr.de/136057/1/aerialmpt_2020.pdf)
describes a 2 Hz processed image sequence. At release stride 1, 7,214 target
windows are history-eligible, 6,344 (87.94%) have all 12 future labels, 792 are
partial and 78 have no future labels. The 61 complete multi-agent rolling
queries reduce to just eight non-overlapping intervals in four conservative
location/event groups. Six clips are shorter than the 20-step grid.

The published tracking split puts both BAUMA and Pasing in train and test.
Exact frame-name intersections across those roles were zero, but that does
not establish independent sites. Grouping by published location/event names
is conservative, not a verified camera or participant identity audit. The
package README lists CC BY-SA 4.0; no source material is redistributed here.
This source has useful dense-image diagnostics, but cannot provide a broad
independent 8-to-12 confirmation by itself.

## OpenDD v3: Better Label Support, Concentrated Sites

I checked all seven [published v3 trajectory exports](https://l3pilot.eu/data/opendd.html):
501 recording tables, 44,914,224 total observations, 1,819 pedestrian tracks
and 1,583 bicycle tracks. All checked timestamps fit the 1001/30000-second
release grid (maximum residual 0.334 microseconds). Stride 12 therefore has
about 0.4004 seconds per sample **for this export only**. This is an audit
cadence, not a revised DroneCrowd or SDD time contract. The publisher supplies
projected map and trajectory coordinates; this run used neither positions nor
the maps, and estimated no cross-dataset physical calibration. The release is
CC BY-ND 4.0 per the publisher; no data or converted trajectory files are
redistributed here.

| Site | Flights | Pedestrian tracks | Eligible targets at stride 12 | Complete 12 | Flights with >=2 complete pedestrians | Non-overlapping complete multi-agent queries |
|---|---:|---:|---:|---:|---:|---:|
| rdb1 | 153 | 963 | 1,741,827 | 1,622,241 | 124 | 1,790 |
| rdb2 | 56 | 16 | 38,073 | 35,769 | 2 | 20 |
| rdb3 | 54 | 38 | 82,836 | 78,798 | 5 | 39 |
| rdb4 | 69 | 50 | 104,641 | 98,234 | 7 | 45 |
| rdb5 | 60 | 122 | 172,744 | 157,794 | 15 | 101 |
| rdb6 | 52 | 607 | 1,217,387 | 1,141,416 | 44 | 1,161 |
| rdb7 | 57 | 23 | 26,768 | 23,594 | 1 | 6 |
| **Total** | **501** | **1,819** | **3,384,276** | **3,157,846** | **198** | **3,162** |

Of eligible pedestrian target windows, 93.31% have complete future labels
(207,575 partial and 18,855 absent). Endpoint and complete-tail counts agree
at stride 12. At stride 1, 3,533,644 of 3,552,838 targets are complete.
At stride 12, rdb1 and rdb6 supply 87.52% of complete target windows and
93.33% of non-overlapping complete multi-agent queries. Seven roundabouts
in two cities are not 501 independent sites, and traffic pedestrians are not
a substitute for dense aerial crowd recordings. Paired past images were not
verified in this intake.

**Next decision:** OpenDD is the stronger candidate for an independently
sourced, leave-location-out trajectory diagnostic against CV and OLS4. A
site-balanced result with per-site support and separate reporting for the two
dominant sites would be more informative than a pooled rolling-window score.
No admission to official confirmation, change to the estimand or claim of
multimodal transfer follows from these counts.

## Reproduction

Use the publisher archives in the ignored directories below. Each JSON
records archive byte counts and SHA-256 hashes. The reader extracts only the
OpenDD trajectory databases into temporary directories and uses read-only
SQLite; existing output is checked byte for byte instead of overwritten.

```sh
out=outputs/publication_readiness_2026_09/topdown_source_intake_v1
.venv-pytorch/bin/python scripts/audit_m3w_topdown_sources.py \
  --aerial external_data/AerialMPT_source_audit/publisher.zip --output "$out/aerialmpt.json"
.venv-pytorch/bin/python scripts/audit_m3w_topdown_sources.py \
  --opendd-dir external_data/OpenDD_source_audit --output "$out/opendd.json"
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_topdown_source_support.py
```

[AerialMPT counts](aerialmpt.json) | [OpenDD counts](opendd.json)
