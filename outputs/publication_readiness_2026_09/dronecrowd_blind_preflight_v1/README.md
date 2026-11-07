# DroneCrowd: Full Input Preflight, Before Choosing Roles

I streamed all 31,472 eligible query times across 112 cached recordings
instead of sampling three prefixes per recording. Every query has a
history-complete forecast target. There are 4,520,912 history-complete
target instances, plus 37,184 current neighbors with incomplete history
that stay in the inputs with masks. The CV tensor check is past-only;
no error, future label or learned prediction was evaluated.

| Prospective option | Role | Exclusion groups | Recordings | Query times | History-complete target instances |
|---|---|---:|---:|---:|---:|
| Separate calibration | Calibration | 15 | 40 | 11,240 | 1,757,584 |
| Separate calibration | Confirmation | 30 | 72 | 20,232 | 2,763,328 |
| Frozen transfer | Confirmation | 45 | 112 | 31,472 | 4,520,912 |

The input-only scan and exact replay each took about eight seconds on
native arm64. It verifies both existing label-blind alternatives without
assigning either role set. The proposed 30-group confirmation's largest
group accounts for 22.98% of its target instances, so rows must not
substitute for independent units or determine the primary group weighting.
Exclusion groups do not certify physical sites; those row counts are
overlapping windows. Easy-reference support was not measured by the
preflight, and cannot be inferred from complete history.

## Predictor Coverage

The existing 18 PCT/EqMotion checkpoint records and two candidate
controller heads have archived source-screen receipts. This turn also
verifies the three frozen self-kinematic checkpoints and warm starts.
Their training-row hash matches the 24-recording, 69,746-row
fitting roster. All 2,688 cached cross-archive image-pair receipts
for that roster are present and unflagged under the original screen.
This is a fresh checkpoint/roster join, not a new image-matching run
or proof of geographical independence. The final external controller
and cost/reference cuts are not frozen.

## Research Decision

I recommend the existing 15/30 option if the main question is whether
an external calibration set makes a fixed intervention rule useful.
The 45-group option tests transport without any external calibration.
Neither changes the original 2%, reference-support or undefined
denominator rules. Author choice and a frozen controller, reference
cut and role manifest are needed before model inference or outcome
readout. Even then, the groups give finite-cohort evidence, not a
guaranteed population failure rate.

The [CVPR 2027 calendar](https://cvpr.thecvf.com/Conferences/2027/Dates)
lists paper registration on November 10, the main submission on
November 16 and supplements on November 23, 2026, all AoE.
The [author guidelines](https://cvpr.thecvf.com/Conferences/2027/AuthorGuidelines)
retain the eight-page main body and mandatory LLM disclosure. The
current draft is not submission-ready. No registration or submission
was made.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_blind_preflight.py scan
.venv-pytorch/bin/python scripts/run_m3w_dronecrowd_blind_preflight.py replay
.venv-pytorch/bin/python scripts/check_m3w_dronecrowd_kinematic_coverage.py
```

The reader uses an offline XML annotation prefix at native stride,
eight observed steps and twelve requested steps. Time and geometry
calibration remain unverified. No model training, external learned
inference, confirmation, deployment, Stage5C or SMC took place.
No new CREATE work was needed for this local scan.
