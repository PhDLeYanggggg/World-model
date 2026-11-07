# Exact Candidate Bank And Reference-Cut Provenance

## Existing Role

The earlier whole-source registry reserves all 112 DroneCrowd
recordings for candidate confirmation. The later alternatives
did not overwrite it. The 45-group zero-shot alternative is
compatible with that reservation; the 15/30 alternative would
move 40 recordings into calibration and requires a role revision.
This audit keeps the original reservation. Neither alternative
is a new predictive admission. The 45 groups are not 45
independently verified physical locations.

## What Was Reconstructed

The eight-alternative bank in the original Moving A policy uses
the fold-2 PCT and EqMotion adaptations, three self-kinematic
fits (one per seed), and five causal forecast alternatives
besides CV. The exact six PCT/EqMotion and three kinematic
checkpoint hashes and cached development prediction references
were checked against the source-screen receipts. CV remains the
risk reference; OLS4 remains the fitting-selected accuracy
comparator. PCT is the project's Past Context Transformer.

The source annotation parser reconstructed the original 195,286
development fitting rows with exact array and inventory hashes.
It reproduced both historical CV-ADE cut pairs using their actual
producer fit rows. The same formula on the exact six comparator
fits yields an additional candidate pair:

| Fitting Provenance | Fit Rows | Easy (pixels) | Hard (pixels) | Status |
|---|---:|---:|---:|---|
| Historical producer 0 | 53,956 | 9.546 | 29.900 | Exact replay |
| Historical producer 1 | 71,584 | 15.404 | 49.249 | Exact replay |
| Matching producer 2 | 69,746 | 12.492 | 61.718 | Candidate only |

The formula is the 25th percentile of positive observed CV ADE
and the 75th percentile of observed CV ADE. The producer-2
manifest declares ten groups, but only nine contribute fit rows
or have normalizers. The additional declared group contributes
zero eligible rows in the reconstructed population; the row hash
for all six checkpoints matches the nine contributing groups.
No external label, prediction or support count was used.

## Remaining To Score

The confirmed predictor provenance and candidate cut do not
freeze the external controller or choose the evaluation cut.
Bind both in a pre-outcome protocol with the one-shot comparison
and source-use disposition. The existing reservation can be
retained for a zero-shot transfer test without moving any
recording into calibration. That test would evaluate empirical
transport, not externally calibrated performance or a guarantee.

The [dataset authors](https://github.com/VisDrone/DroneCrowd)
provide the full release and request citation. This audit does
not infer broader redistribution rights from its public availability;
the source data remain local. Image pixels and raw annotation
steps are not calibrated distance or time. The original 2%
evaluation rule and the Stage5C and SMC settings remain unchanged.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_exact_bank.py audit
.venv-pytorch/bin/python scripts/audit_m3w_dronecrowd_exact_bank.py replay
```

The reconstruction uses only admitted local development
annotations and previously checked forecast and audit metadata.
No new training or CREATE workload. The [analysis](analysis.json)
contains exact cut values and source/checkpoint hashes.
