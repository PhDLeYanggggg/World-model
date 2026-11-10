# Athens: score the main comparison and attribution in one pass

The primary procedure is unchanged. The [frozen recipient/path bank](../external_recipient_factorial_v1/README.md)
adds a separate secondary readout, using the same member targets
already joined by the primary procedure. Its observed components
must add to the diagonal contrast. The diagonal itself must reproduce
the primary point and missing-label envelope at every seed.
No component is used for calibration or release decisions.

I rehearsed both protocols on all the actual forecast packets with
fabricated futures. The four-date zero-shot version made six held
member joins and zero calibration calls. The chronological version
made eight calls across the four calibration members and exactly one
join for each of the two held members. The two summaries rebuilt
identically from the saved member scalars without another target
call. No values from fabricated labels are exported.

The [receipt](rehearsal.json) is an engineering check, not an Athens
result. Neither date assignment has been registered for real labels.
After the author chooses, use the integrated runner with the approved
protocol; it verifies publication before any target access.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/rehearse_m3w_pneuma_confirmation_with_attribution.py
```

[Attribution definition and bounds](protocol.md).
