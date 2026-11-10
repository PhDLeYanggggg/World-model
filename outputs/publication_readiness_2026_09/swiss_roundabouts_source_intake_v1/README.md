# Swiss roundabouts: a second unscored vehicle source

I cached all 18 trajectory CSVs and the two documentation files from the
[EPFL publisher release](https://zenodo.org/records/15077435).
Every member matches the published byte count and MD5; the
[source receipt](source.json) also records SHA-256. The archive is
CC BY 4.0. No videos, raw trajectory rows or fitted weights are
included here.

## Clock and support

The publisher describes 29.97 FPS acquisition. The released
centisecond timestamps fit a 30 Hz lattice within uninterrupted
segments. A single phase fails for the whole of `D2_AM2_L1.csv`.
I split all files at gaps longer than four centiseconds, yielding 31
segments. Every segment then fits the quantized grid with at most
6,667 microseconds of residual width. No interpolation or windows
across segment boundaries. The pilot stride is 12 release ticks; it is
not a physical-time verification or a registered evaluation horizon.

| Roundabout | Files | File-local track IDs | 8/12 queries | Eligible targets | Complete targets | Complete rosters | Observed pair-steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frick F1 | 8 | 4,134 | 1,112 | 17,888 | 15,579 | 128 | 1,688,965 / 1,932,168 |
| Frick F2 | 6 | 2,521 | 864 | 5,632 | 4,212 | 127 | 159,911 / 214,116 |
| Laufenburg L1 | 4 | 2,130 | 587 | 4,150 | 2,893 | 49 | 142,263 / 196,812 |
| Total | 18 | 8,785 | 2,563 | 27,670 | 22,684 | 304 | 1,991,139 / 2,343,096 |

The 7,386,908 rows contain 8,785 observed file-local IDs; these are
counts from the released CSVs, not the publisher's campaign totals
or a count of unique people across files. For each grid I select the
earliest nonoverlapping windows with at least two agents present
at all eight historical ticks. Future token presence is counted
only after the roster is fixed. At stride 12, 84.98% of pair-steps
have both coordinate tokens, but only 304 windows have complete
futures for the full roster. A complete-case-only readout would
discard most windows.

## Observed inputs and frozen forecasts

I rebuilt the same 2,563 rosters and decoded numeric positions only
at the eight planned history ticks. Each window uses a fixed WGS84
tangent-plane transform, with local translation from the first roster
member's last observed point. This supplies a working coordinate frame, not a check of the
publisher's georeferencing accuracy. The original
[Geo-trax v0.1.0 code](https://github.com/rfonod/geo-trax/blob/3d60ffa1c8f16e42d65de12054dc161ceb0de716/process_video.py)
postprocesses vehicle class over the whole track, so I did not use that
field as an input. This verifies our history access, not how every
publisher-exported position was constructed.

The [prefix receipt](prefix_receipt.json) binds each observed packet.
The [forecast receipt](forecast_receipt.json) covers CV, OLS4,
median velocity and the frozen OpenDD vehicle forecaster, with all
21 donor checkpoints verified across three seeds. I did not fit on
Swiss data, join future coordinates or evaluate accuracy.

The [matched proposal receipt](matched_receipt.json) freezes four arms:
independent, additive, same-family and unrestricted joint. Each arm
changes exactly two distinct recipients at the same maximum radius.
Of 7,689 query-seed views, 7,431 have a legal pair for all four arms;
258 have only two actors and fall below the inherited three-actor
menu minimum. They remain in the full roster, as verified by the
[readout rehearsal](../swiss_roundabouts_readout_rehearsal_v1/README.md).
Joint selects a different proposal
from independent, additive and same-family in 5,157, 5,074 and
1,371 comparable views, respectively. That is proposal diversity, not
accuracy or utility. The 54 local file-seed packets are resumable.

## What this source can establish

The three roundabouts were filmed on one day. Two are in the same
municipality, and six file pairs overlap in time across different
drones. Treat the files as repeated observations within a campaign,
not 18 independent sites. This is a useful additional roundabout
transfer cohort, but it does not replace the four-date Athens
comparison. Neither scientific roles nor thresholds were assigned.
The 2% rule, existing release decisions and v191 manuscript are
unchanged.

[Structural coverage](support.json) | [Source receipt](source.json)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/fetch_m3w_swiss_roundabouts.py
.venv-pytorch/bin/python scripts/audit_m3w_swiss_roundabouts.py
.venv-pytorch/bin/python scripts/build_m3w_swiss_roundabout_pilot.py prefix
.venv-pytorch/bin/python scripts/build_m3w_swiss_roundabout_pilot.py forecast
.venv-pytorch/bin/python scripts/build_m3w_swiss_roundabout_proposals.py
```
