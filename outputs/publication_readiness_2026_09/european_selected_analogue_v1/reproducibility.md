# Reproducibility and Evidence Scope

## Observed Execution

Registration commit1f0b3250 preceded the real pilot and full run. PilotPID71668
exited0,20.230784s total,10.635752s view,2,426,093,568 bytes peak RSS. All6,709
pilot queries had references; this pilot was not a scientific screen.

Full runPID71944 exited0 with72/72 immutable view receipts,160.680342s and
2,915,418,112 bytes peak RSS. This was nonparametric fitting/retrieval and frozen
Torch inference, **zero neural optimizer updates**, not neural training.

Full replayPID72471 exited0 at2026-10-29T04:15:11.633139Z. It reloaded source
packets and frozen models, reran all72 reference retrievals and reproduced every
group and aggregate. Complete artifact SHA256:
`31adb79c2d646961e25cd100d09aa4b44b22702ed5078686c42d30e303a41add`.
Registration SHA256:
`fe3487888b5133918d5bce3ba505d58493027b47b3f3bbecd8f14aa3f400efda`.

Separate arithmetic verifies72 views,720 arm-target records,443 recording banks,
144 prior-control cohort comparisons and all3 descriptive screen conditions.
Twenty-one descriptive slices and the figure hash replay. Twenty-four scoped
tests pass3.35s, including target-record-label intervention, unknown/support
retention, deterministic retrieval and corrupted-receipt rejection. The local
figure was visually inspected; it displays shared-support diagnostics only.
These checks are not independent external replication or scientific gate passes.

## Verified Commands

Run from the project root using native arm64 `.venv-pytorch/bin/python`.
CPU4, inter-op1, single process; no DataLoader multiprocessing, MPS or new HPC job.
The existing authorized original-TRAIN packet route is reused. No claim is made
about the current CREATE scheduler or unrelated jobs from successful data reads.

```bash
.venv-pytorch/bin/python -m scripts.run_m3w_selected_analogue check
.venv-pytorch/bin/python -m scripts.verify_m3w_selected_analogue --check
.venv-pytorch/bin/python -m scripts.report_m3w_selected_analogue --check
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_selected_analogue.py tests/test_m3w_selected_analogue_verifier.py tests/test_m3w_selected_analogue_report.py tests/test_m3w_risk_decoupling.py tests/test_m3w_risk_decoupling_verifier.py tests/test_m3w_query_context_risk.py tests/test_m3w_query_context_readout.py
```

For a new environment with authorized source packets and exact parent assets:
register and commit the bound files, runpilot, thenrun. Never delete completed
receipts to rerun. Use `run --resume` only after an interrupted, incomplete run;
first inspect the saved PID, lock and heartbeat. Complete phases are immutable.
Each view is an atomic checkpoint; a partial view is recomputed on resume.
Disk reserve10GiB and12h execution boundary remain. No reduced scientific scope.

Private log and figure: `data/stage_cvpr2027_experiments/european_selected_analogue_v1/`.
Only code, protocol, aggregate receipts, reports and light metadata enter Git.
The unrelated staged index fingerprint remains
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

## Scientific Limits

No held forward pass, independent role, threshold search, policy intervention,
new trajectory training, Stage5C or SMC. All72 views completed but48 lack donor
support; completion is not a positive scientific result. Twelve exposed TRAIN
source localities, repeated occurrences and upstream exposure are not external
confirmation. No metric/seconds, human gold, physical safety, true3D or foundation
claim. Full legacy artifact-writing tests were not rerun; scoped tests do not
stand in for the eventual independent research evidence.
