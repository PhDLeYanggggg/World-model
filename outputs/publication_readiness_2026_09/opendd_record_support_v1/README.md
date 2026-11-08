# Training on Whole-Recording Support

I tested whether second-edit selection improves when the label is
the outcome of **both edits on the full recording**, rather than
the outcome of the small edit on its own. The 598 first edits,
345 second slots, second query and amplitudes (0.25 and 0.10)
are fixed on the evaluation roster. This compares recipient
choice only, not how many interventions to make.

## Nested Training

The seven OpenDD sites are reused development evidence. For a
held site and a donor site, the first-edit scorer fits on the five
remaining sites. The trajectory producing each of those fitting
examples excludes the held site, donor site, and its own site.
I trained all 35 triple-exclusion partitions from scratch for each
of three seeds, then 126 first-edit heads and 63 second-edit heads.
The outer evaluation reuses the original held-site trajectory
and first-edit heads. No thresholds are selected from outer scores.

The second-edit model sees the current menu and the first edit
only if it occurs at or before that query. The full recording
is used only to create training labels and score finished
predictions. The local-label control receives the same feature
rows, training budgets and first-edit context. The masked
control trains on the same recording labels without that context.
The finite label calculation is checked against the original risk
and conservative two-reference scorer. Donor cells have only
57 to 176 matching recording menus (157 to 407 candidate rows);
105 forecast fits do not turn these into more independent recordings.

## Results

| Policy | Second recipients changed | Undefined | Risk pass | Joint pass |
| --- | ---: | ---: | ---: | ---: |
| Frozen easy | 0 | 295 | 172 | 138 |
| Local labels, same data | 67 | 301 | 160 | 126 |
| Recording labels, no first-edit context | 149 | 312 | 157 | 124 |
| Recording labels and first-edit context | 115 | 314 | 155 | 123 |
| Guarded recording head | 83 | 313 | 153 | 120 |

The recording head improves conservative error by +0.00498 pp
[+0.00082, +0.01009] versus frozen easy, but loses 15 joint passes
(11 at one site and four at another). It gives **+0.00328 pp
[-0.00154, +0.00896]** against the matched local-label control.
The context ablation gives +0.00371 pp [-0.00070, +0.00928]
for including the first edit. Neither comparison establishes a
reliable advantage of the new training labels or context.
The guard gives +0.00251 pp [+0.00020, +0.00582] versus easy,
but loses 18 joint passes. Its predicted risk comparison is not
a certificate of the realized recording risk.

Intervals are 95% site bootstraps with 3,000 draws, averaging the
three seeds before resampling. The 606 views are repeated scores
for 202 recordings, not 606 independent scenes. The exact previous
[same-clock bound](../opendd_support_stop_v1/README.md) is 178/606
even with outcome-aware recipient choice. None of these policies
passes the unchanged full-roster rule.

This ends the case for another small reranker at the same clock.
The next defensible change is to redesign the intervention
schedule on development data around available reference support,
with a matched sparse control, and check which recordings have
no reference support even with a larger candidate menu. More
compute at the same clock cannot remove that ceiling.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_nested_support_bank.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_record_support.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_record_support.py --replay
.venv-pytorch/bin/python scripts/freeze_m3w_dronecrowd_calibration_roles.py verify
```

`fresh_run`: 294 fits and 178,500 optimizer updates on local
arm64 Torch, four threads and no data-loader workers. The trajectory
bank took about 23 minutes; the heads took about one minute.
Both runners resume from the latest checkpoint and update heartbeat
every 100 steps. `cached_verified`: original outer forecasts and
the recorded checkpoints for replay. `not_run`: independent
DroneCrowd confirmation. The 15/30 split and original 2% rule are
unchanged. No deployment, Stage5C, SMC, or paper submission.
