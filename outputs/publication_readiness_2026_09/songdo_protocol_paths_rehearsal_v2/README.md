# Songdo: two prospective evaluation paths

The same four frozen intervention arms and two-edit budget work under
both options below. I have not assigned either option to real labels.

| Option | Calibration roster | Evaluation roster | What it answers |
| --- | --- | --- | --- |
| Calibration plus independent confirmation | A/B, C and E: 3 linked components, 40 flights, 31 with matched proposals | 14 remaining components, 160 flights, all with matched proposals | Can a policy calibrated on disjoint sources improve forecasts while meeting the original risk checks? |
| Matched-only zero-shot | None | 17 components, 200 flights, 191 with matched proposals | Does the frozen rule improve utility at the same edit count and radius, without external calibration? |

The calibrated path uses the existing OpenDD cutoff convention:
pool positive CV ADE on complete 12-step actor paths from calibration
flights only, then use the 25th and 75th percentiles in the dataset's
native coordinates. The runner freezes those cutoffs and fits the
tie-complete passing score prefix for each arm and seed on those same
calibration flights. Confirmation receives the frozen values and
cannot refit them. The original 2% checks are unchanged.

The matched-only path has no cutoff fitting or threshold. It compares
the four arms at the same two edits and radius, with CV and OLS4
references, and reports native errors, missing-step conservative
contrasts, and relative motion. It does **not** compute release
coverage, risk passes or deployment certification. Missing proposal
flights are counted, not treated as successful edits.

## Reproduce

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/rehearse_m3w_songdo_confirmation.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_confirmation.py tests/test_m3w_songdo_matched_readout.py tests/test_m3w_songdo_future_join.py
```

Both paths ran on the real frozen predictions with fabricated
future displacements and masks. The calibrated rehearsal scored
480 confirmation seed views and generated cutoff receipts only for
the 40 calibration flights. The zero-shot rehearsal scored 600 seed
views, of which 573 were matched. The [receipt](receipt.json)
contains no effects from invented labels. Thirty-nine Songdo tests
passed. No real label index or coordinates were opened.

For a real run, the manifest needs explicit approval, the complete
site-to-role assignment, and the declared method. Use
`cutoff_rule: calibration_complete_positive_cv_ade_q25_q75` with
`release.method: risk_passing_score_prefix` for the calibrated path,
or `cutoff_rule: not_used` with `release.method: matched_only`
for zero-shot utility. Then use the existing
[register/calibrate/confirm commands](../songdo_confirmation_rehearsal_v1/README.md#real-run-after-protocol-approval).
The same command resumes from per-flight checkpoints. No production
manifest or release is included in this rehearsal.
