# Songdo: secondary readout without reopening outcomes

I added a reporting step that runs after the primary confirmation,
using its existing per-flight score receipts. It checks the original
registration, release freeze, complete flight roster and three seeds.
It replays the primary summary and verifies that every recorded
release matches its frozen threshold before writing a new report.
It cannot read dataset archives, refit cutoffs or open labels.

| Scope | Reported measures |
| --- | --- |
| Same matched flight subset | CV, OLS4 and all four arms; available and complete-roster ADE/FDE, flight p95 ADE, intervention rate and pairwise contrasts |
| Full-roster motion references | CV and OLS4, including flights without a matched proposal |
| Released policy, if registered | Accuracy including CV fallback, original risk support, dual-reference passes and counts by linked source |
| Edited views only | Defined signed easy degradation, easy and hard available-label gains, and explicit support counts |

The report separates the worst estimable source mean from the worst
single flight-seed view. Tail values are source summaries of flight
p95s, not a pooled trajectory percentile. It uses the original
3,000 source-level resamples after averaging seeds within flight
and flights within source. No-edit views contribute to accuracy but
do not certify risk. Unsupported easy or hard quantities stay
undefined, and matched-only runs have no release section.

## Rehearsal

| Fabricated-target path | Flights | Linked components | Flight-seed views | Matched views |
| --- | ---: | ---: | ---: | ---: |
| Calibrated confirmation | 160 | 14 | 480 | 480 |
| Matched-only zero-shot | 200 | 17 | 600 | 573 |

Both paths used the real frozen proposals with fabricated futures
in temporary run directories. Both reproduced the primary summary.
Only structural counts were exported. Twenty focused tests passed.
This is `fresh_run` for reporting and integration, `cached_verified`
for the forecast files, and `not_run` for scientific confirmation.
No production roles or outcome coordinates were read.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/rehearse_m3w_songdo_secondary.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_secondary.py tests/test_m3w_songdo_confirmation.py tests/test_m3w_songdo_matched_readout.py tests/test_m3w_songdo_future_join.py
# Only after the approved primary confirmation finishes:
.venv-pytorch/bin/python scripts/report_m3w_songdo_secondary.py /path/to/completed/run
```

The standalone command writes `secondary_report.json` alongside the
existing results; it does not alter `confirmation.json`. I retain the
[two pending protocol choices](../songdo_protocol_paths_rehearsal_v2/README.md).
[Rehearsal receipt](receipt.json).
