# Songdo: the complete first day, ready for a prospective evaluation

I expanded the original four-site sample to all 20 first-day
intersections in [Songdo Traffic v2](https://zenodo.org/records/17924857).
All 200 flight CSVs are represented. The 20 archives total
2,787,359,162 bytes; publisher MD5 and size match, and each archive has
a local SHA-256. I used the complete first day, not a subset chosen by
model performance. The underlying data are described by
[Fonod et al.](https://doi.org/10.1016/j.trc.2025.105205).

## Coverage on the unchanged 8/12 native grid

| Structural count | A/B/C/E, verified reuse | Other 16 sites, new | Total |
| --- | ---: | ---: | ---: |
| Flight CSVs | 40 | 160 | 200 |
| Raw metadata rows | 27,778,328 | 150,020,197 | 177,798,525 |
| Nonoverlapping windows | 1,992 | 12,284 | 14,276 |
| Sampled agent histories | 81,000 | 547,902 | 628,902 |
| Windows with at least one complete pair | 1,991 | 12,254 | 14,245 |
| Windows with complete future for the full roster | 124 | 1,055 | 1,179 |

716 of 758 drone fragments support the original grid and visibility
screen. The observed pack contains only coordinates from the eight
past steps. The label index records future row pointers and
availability, not future coordinate values. Complete pair coverage is
13,887,247 / 17,255,540 among sampled windows. Only 8.26% of windows
have complete future for every observed actor. Pair-level and
whole-roster results must therefore stay separate.

## Candidate source groups

A/B, F/G and K/L overlap both in the publisher's map cutouts and in
the sampled observed spatial envelopes. I keep each pair together.
The recorded metadata show no same-drone cross-site overlapping
clock spans or matching CRC/size members. Across the three
spatially overlapping pairs I found no exact cross-site sampled
prefix hashes and no near-synchronous sampled matches. The 20
sites form 17 candidate source components. This checks observed
duplication and capture overlap; it cannot rule out a vehicle
travelling between sites at different times. The components are
not yet calibration or confirmation groups.

## Unscored vehicle predictions

I generated CV, OLS4, median4 and OpenDD vehicle-completion predictions
for all 628,902 histories. For each of three seeds I average the seven
verified location-excluded OpenDD checkpoints. This bank supersedes
the pedestrian bank as the task-matched vehicle input, not as an
evaluated model. No new training was run. At the existing equal-radius
budget, 14,266 / 14,276 windows support both the joint and same-family
two-edit options for each seed. This counts legal actions, not model
accuracy or risk passes. The bank did not open the label index.

**Next:** fix the prospective Songdo roles before reading future
coordinates or scores. Compare CV/OLS4 and matched intervention rules
on the same label masks and report uncertainty by source group. The
original 2% rule and the sealed DroneCrowd confirmation are unchanged.
No deployment, Stage5C, SMC, or manuscript change.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv-pytorch/bin/python scripts/fetch_m3w_songdo_first_day.py
.venv-pytorch/bin/python scripts/audit_m3w_songdo_first_day.py
OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/build_m3w_songdo_first_day_vehicle_bank.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_first_day.py tests/test_m3w_songdo_publisher_footprints.py tests/test_m3w_songdo_prefix_pack.py tests/test_m3w_songdo_traffic.py tests/test_m3w_songdo_candidate_bank.py tests/test_m3w_vehicle_residual.py
```

The audit uses a single local process, and inference uses four
compute threads without data workers. Acquisition can resume
partial archives, and the audit and forecast bank checkpoint
each flight. Twenty-four focused tests passed. `fresh_run`: 16
site audits, first-day linkage and vehicle inference.
`cached_verified`: four earlier site packs and all 21 donor
weights. `not_run`: external scoring and policy selection.
Raw files, label indices and forecasts stay local.

[Protocol](protocol.md) | [Archive checksums](archives.json) |
[Input and coverage](input_receipt.json) | [Linkage](lineage.json) |
[Unscored bank](vehicle_bank.json).
