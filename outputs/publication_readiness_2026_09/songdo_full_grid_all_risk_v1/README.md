# Songdo: all-edited risk on the full twelve-step grid

The original 2% rule averages over observed steps for a partially labeled
track. It is not a guarantee about full-grid deterioration. I replayed its
all-edited ratio for each matched view, then calculated a separate exact
worst-case all-edited ratio on twelve steps. For every missing point, setting
the target to CV minimizes its CV error and maximizes its forecast-minus-CV
error. After summing within a track and taking positive deterioration,
this maximizes the ratio when observed CV mass is positive. The argument
does not certify a full-grid easy subset, physical plausibility, or any
cross-seed common completion.

| Matched rule | Three original checks pass / 480 | Those also passing full-grid all-edited | Those with an adverse full-grid completion |
|---|---:|---:|---:|
| Independent | 179 | 176 | 3 |
| Additive | 173 | 170 | 3 |
| Same family | 71 | 70 | 1 |
| Joint | 69 | 68 | 1 |

The adverse completion makes the diagnostic ratio exceed 2%; it is not
evidence that the actual missing labels produce a violation. The original
matched proposals and the actual released policy are different populations.
The only actual releases were 15 independent-policy views. All selected
tracks in those views had twelve labels. Fourteen pass the original and
full-grid all-edited check, but only five pass all three original checks.
The joint arm still releases zero. No abstention is counted as a pass.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_songdo_full_grid_risk.py
```

`fresh_run`: the new all-edited sensitivity, using numeric labels only from
the **previously scored** 160 confirmation flights. `cached_verified`:
the matched proposals, original risk counts, and release roster. `not_run`:
new calibration, new independent outcomes, training or deployment changes.
[Protocol](protocol.md) | [Aggregate readout](readout.json).
