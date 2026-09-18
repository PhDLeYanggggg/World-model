# Reproducing the Fixed Deferral Readout

## Contract

Run from the repository root `/Users/yangyue/Downloads/World`. Use the native
arm64 `.venv-pytorch/bin/python`, not the historical Intel Conda environment.
This is inference from already trained, locally available checkpoints. Do not
restart training to reproduce this readout, change threshold 0, select a seed,
or score sealed main roles. Missing or changed bound inputs must fail closed.

Registration commit: `92b96b78`, made before source inference.
Registration SHA256:
`9d4b3c7d35eaa4a679da06ce59678ef523654f02078c19a793d5be65075b6ff2`.
The registration binds the runner, statistical consumer, prior reports and
decision. Checkpoint/data bindings appear in `frozen_predictors.json` and
`evaluation.json`; per-row predictions remain private.

## Commands Actually Run

```bash
.venv-pytorch/bin/python scripts/run_m3w_source_deferral_transfer.py --registration configs/m3w_source_deferral_transfer_v1.json --audit-only
.venv-pytorch/bin/python scripts/run_m3w_source_deferral_transfer.py --registration configs/m3w_source_deferral_transfer_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_deferral_transfer.py --registration configs/m3w_source_deferral_transfer_v1.json --replay
.venv-pytorch/bin/python scripts/verify_m3w_source_deferral_transfer.py --registration configs/m3w_source_deferral_transfer_v1.json
.venv-pytorch/bin/python scripts/diagnose_m3w_source_input_collisions.py --training-registration configs/m3w_source_cost_deferral_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_input_collisions.py tests/test_m3w_deferral_transfer.py tests/test_m3w_recording_diagnostic.py tests/test_m3w_source_cost_deferral.py
```

The last diagnostic is explicitly post-hoc and training-only, not part of the
preregistered forecast comparison. Twenty-two focused tests pass. Full legacy
tests were not rerun; earlier test counts refer to their own fixed versions.

## Run Receipts

| Operation | PID or session | Outcome |
| --- | --- | --- |
| Input audit | PID 45838 | exit 0 |
| Source inference | PID 45964 | exit 0; 33.8142 second log span |
| Exact replay | PID 46208 | exit 0; 23.5873 second log span |
| Independent verifier | session 99619; repeated-eval child PID 46454 | exit 0; zero training updates |
| Initial exact-input diagnostic | session 45808 | exit 1 at JSON serialization; log retained |
| Fixed diagnostic retry | session 81328 | exit 0; prior fingerprints verified |

No active process remains from these runs. Local execution is sufficient for
this readout. No CREATE job was submitted or newly observed; current remote
assets/jobs remain unknown, not assumed absent from an old access failure.

The verifier recomputes 15 per-seed output cells and five three-seed summaries,
then repeats evaluation. All 313 inventoried artifacts remain unchanged.
Replay checks six neural proposal/score arrays and three cached dense arrays
exactly. All fresh arrays are finite, bounded and row-aligned; fallback and
unsupported outputs are exact baseline. The two-panel figure was visually
checked. Fontconfig/Matplotlib external-cache warnings were nonfatal; no
numerical or image-generation failure occurred.

## Repair Disclosure

The first training-input diagnostic completed its fingerprint file but failed
when a NumPy-derived integer reached JSON serialization. A standard-JSON
regression assertion reproduced the error. Converting the sufficient-condition
indicator to a built-in boolean fixed it. The retry compared saved fingerprints
exactly; no sample, label, grouping, model or threshold changed. Both logs are
retained privately as `input_collision.log` and `input_collision_retry.log`.

## Verified Outputs

| Artifact | SHA256 |
| --- | --- |
| evaluation.json | 06ac38de9c15db90a60264ac17d79f05f39241c81ac06d7c8141b52a1fc35667 |
| verification.json | 3695ca7e955de4f58ddc0c01c4bb96060983d241774dfe7f79a7165c49a507e7 |
| Private training-input fingerprints | e388e8f98fb715fab81b1c94bd46bc5798f258cd82256a90868429bd60ca7ca4 |

The bootstrap uses 2,000 paired recording resamples, seed 38113. These are
conditional diagnostics for one historically explored site. Seed-mean errors
are not an ensemble. Easy relative harm is undefined when CV error is zero.
Engineering reproducibility does not establish positive transfer or novelty.

Only code, reports, light aggregate metrics and the original summary SVG are
eligible for Git. Do not add raw data, per-row fingerprints/predictions,
checkpoints, third-party imagery, caches or `.venv-pytorch`. Stage5C/SMC remain
off; raw-frame and pixel/past-normalized claims only.
