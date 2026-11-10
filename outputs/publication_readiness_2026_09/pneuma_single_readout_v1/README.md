# Athens: one readout for the main comparison and both controls

The same future array and validity mask now feed the original
peak-matched comparison, the recipient/path attribution, and the
step-matched utility sensitivity. The latter reports the original,
uniformly smaller-peak, and equal-step paths on the same query
roster. It does not set thresholds or issue a release decision.

The driver checks the exact 18 frozen forecast/profile packets before
joining targets. It refuses to score before publication of the original
registration and operating-rule freeze. It writes one scalar receipt
per confirmation member and reconstructs the three reports from those
receipts. The frozen primary scores are checked against the original
variant in the secondary analysis.

The [rehearsal](rehearsal.json) uses the actual frozen Athens forecasts
and fabricated future positions only. It runs both prospective paths:
four dates with donor-only cutoffs, and two earlier dates for calibration
followed by two held dates. It exercises missing-publication rejection,
resumption after one completed confirmation member, and exact rebuilding
of the three reports without another target join. The original primary
metrics and release outputs are compared with the primary-only runner.
No fabricated effect estimates are exported.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/rehearse_m3w_pneuma_single_readout.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_pneuma_single_readout.py tests/test_m3w_pneuma_confirmation.py tests/test_m3w_pneuma_future_join.py tests/test_m3w_pneuma_proposal_separation.py tests/test_m3w_external_step_budget.py tests/test_m3w_two_factor_bounds.py tests/test_m3w_step_budget_match.py
```

Formal phases use `scripts/run_m3w_pneuma_single_readout.py` and the
author-approved protocol: `register`, `calibrate` only if selected, and
`confirm`, each with `--protocol`. This rehearsal does not pick a
protocol or open any real futures. The Swiss forecasts and manuscript
v213 are unchanged.

[Protocol](protocol.md) |
[Frozen movement profiles](../external_step_budget_v1/README.md) |
[Attribution](../pneuma_recipient_attribution_v1/README.md).
