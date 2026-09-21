# Scene-Context Reproduction

Run locally in the verified native arm64 environment. This reconstruction and
scalar verification do not train a model or require CREATE; no fresh remote
scheduler state is claimed. Four CPU threads, inter-op1 and no workers are used
by the runners. Only33admitted source recordings are read; original validation,
test, main, external and bookstore roles remain closed.

```sh
.venv-pytorch/bin/python scripts/audit_m3w_native_scene_alignment.py
.venv-pytorch/bin/python scripts/build_m3w_native_scene_context.py
.venv-pytorch/bin/python scripts/verify_m3w_native_scene_context.py
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_scene_alignment.py tests/test_m3w_native_scene_context.py tests/test_m3w_native_scene_context_verification.py
```

The support audit intentionally reads existing labels to describe missingness;
the context builder and independent verifier never load future target arrays.
All selection choices are frozen. No training, risk calibration, threshold
search or new outcome-based row selection occurs.

Audit session76630/PID79511 exited0 in7.80seconds. Its analysis SHA256 is
`8a5879cfc511943bb303e653e7a5aac0b8b3434792eaaee74c48e1bb5fbdee7f`.
The first context attempt completed but inherited unused label arrays, so its
target-read metadata was rejected. Final past-only v2 session20922/PID79877
exited0 in10.78seconds. Replay40810/PID80081 exited0 in1.65seconds and reproduced
the analysis SHA256:
`ea58e42239180cd326cac5c22294b3fbf1668c484764d212d504bb400a99c17e`.

The first scalar verifier used an exact equality check for mathematically
equivalent arithmetic orders and stopped at a2.27e-13pixel difference. The
verifier now records a numeric QA tolerance, not a safety margin; session41522
exited0 and verified all links. Maximum CV arithmetic discrepancy is4.55e-13.
The builder, frozen model code and risk criterion were not changed for this
numerical check.

The16new tests pass. The expanded139-test set, including the native forecast,
nested cost lineage, metrics, gain/harm, protected risk, geometric risk and
experiment contracts, passes in2.67seconds (session25494). This is a scoped
regression result, not the entire historical repository suite.

Private context caches total219,663,642bytes under
`data/stage_cvpr2027_experiments/native_scene_context_v2/`. Each recording has an
atomic cache and receipt; rerunning verifies completed records and resumes any
unfinished recording. The input-only alignment index remains private under
`data/stage_cvpr2027_experiments/native_scene_alignment_v1/`. Its positional
neighbor links failed and must not be used as identities; use v2contextIDs.
No raw annotation, image, history cache, model weight or row-level data is in Git.

Bound code and arrays must not be overwritten for a new scientific comparison.
Start a new registered version for joint selection. Keep the strict zero-CV
criterion, declared metric and known-versus-unknown outcome distinction intact.
