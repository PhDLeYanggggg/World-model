# pNEUMA: independent-source intake before any prediction scoring

I screened the Athens vehicle trajectories to see whether a new collection
could resolve the relative-motion ambiguity left by the OpenDD vehicle
ablation. The data come from the [pNEUMA release](https://doi.org/10.5281/zenodo.10491409),
described by [Barmpounakis and Geroliminis](https://doi.org/10.1016/j.trc.2019.11.023).
Data source: pNEUMA – open-traffic.epfl.ch. The repository retains the
catalog, code and counts, not trajectory files.

## Publisher intake

- The Zenodo ZIP catalog exposes 219 named, nonempty vehicle CSVs across
  four dates, 20 half-hour session blocks and ten individual drone labels.
  The all-drone exports are alternative views of sessions, not extra
  independent recordings.
- The download form returned an empty response for files that the archived
  release lists. Exact HTTP ranges from the pinned archive retrieved the
  catalog and six members without downloading the 15.76 GB archive.
  Member CRCs and local SHA-256 hashes are in [samples.json](samples.json);
  the full archive checksum was not independently computed.
- The pilot uses the same D1 time block on all four dates and adds D3 and
  D10 on October 29. Six CSVs provide 7,990 file-local tracks and
  13,555,115 timed observations. The publisher coordinates are WGS84;
  no projection or positional accuracy check was performed.

## Support without forecasting

The observed timestamps in these files fit a 0.04-second grid without
residuals above one microsecond. In this pilot the within-track intervals
are all 0.04 seconds; that need not hold in uninspected sessions. For
each stride I take the earliest nonoverlapping query with eight past
samples and at least two vehicles, then measure support across the
entire past-defined roster. A vehicle does not have to survive to the
last future step to be selected. The audit reads future timestamps and
whether position fields are present, not their numerical values; it
decodes only the first eight observed positions per track for a
structural linkage screen.

| Native ticks per step | Past-selected queries | Pair-steps with both position fields | Total pair-steps | Field presence |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 6,163 | 598,441,958 | 601,891,800 | 99.43% |
| 5 | 1,282 | 117,261,415 | 121,454,700 | 96.55% |
| 10 | 643 | 54,558,888 | 58,594,056 | 93.11% |

At stride 10 only 20 of 643 entire rosters have all positions present
for all twelve steps. Even dense per-pair coverage does not justify
a complete-case evaluation of whole scenes. Pair-step counts are not
independent samples or validation of coordinate accuracy. No clock
has been chosen for the predictive experiment, and the release clock
does not change the interpretation of raw-frame horizons on SDD.

## What counts as an independent source?

All pilot members are from one Athens collection. The four D1 dates
cover the same location. On the simultaneous October 29 files, the
first-position D3 and D10 boxes intersect; no intersection with the
D1 box was detected. The screen compares rounded first-eight prefixes
and found no exact duplicate prefixes. That cannot exclude a vehicle
continuing across adjacent drone views. Numeric track IDs recur across
files and are not global identities. Treat same-session views as linked
until a stronger linkage audit exists; neither six files nor 643
windows are six or 643 independent source groups.

The quickest informative next test is a frozen zero-shot comparison
on a previously unused acquisition campaign. Do not train or tune on
Athens outcomes. Keep 8 observed/12 predicted steps and the original
2% criterion; report per-date effects and support, and do not treat
four dates in one city as cross-city replication. Using Athens
for calibration instead would answer a different question and reduce
the number of untouched dates available for evaluation. Predictive
roles, the clock and the exact outer roster are not assigned by
this intake. No benchmark or deployment claim is supported yet.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/intake_m3w_pneuma.py catalog
.venv-pytorch/bin/python scripts/intake_m3w_pneuma.py sample \
  20181024_d1_0830_0900.csv 20181029_d1_0830_0900.csv \
  20181030_d1_0830_0900.csv 20181101_d1_0830_0900.csv \
  20181029_d3_0830_0900.csv 20181029_d10_0830_0900.csv
env OPENBLAS_NUM_THREADS=1 .venv-pytorch/bin/python scripts/audit_m3w_pneuma_support.py
```

The commands verify the publisher metadata, reuse verified local
members and rebuild [support.json](support.json) without model weights.
No CREATE job, training, future-coordinate join or forecast scoring was
needed. The two focused tests check that an absent future token does
not determine the chosen query and that off-grid or duplicate
identities do not silently enter the structural clock.
