# Reproduction

Use native arm64 .venv-pytorch/bin/python. All phases use a process lock,
CPU4/interop1/workers0, checkpoints/heartbeat, a 5 GiB disk reserve and a
512 MiB private artifact cap.

```sh
.venv-pytorch/bin/python scripts/diagnose_m3w_moving_admission_loss.py
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py register
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py pilot
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py train
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py evaluate
.venv-pytorch/bin/python scripts/run_m3w_moving_admission_priority.py verify
.venv-pytorch/bin/python scripts/report_m3w_moving_admission_priority.py
```

Completed phases are immutable. Do not rerun them to overwrite results.
Interrupted fitting uses the same phase with --resume. Private prerequisites
must match recorded hashes. Reproduce in a separate checkout/artifact namespace,
not by deleting the original valid checkpoints. Existing verified reports can
be rechecked with the unchanged reporter; it rejects output drift.
The first registration is preserved; registration_v2 is operative after the
documented synthetic-fixture amendment before real training.
