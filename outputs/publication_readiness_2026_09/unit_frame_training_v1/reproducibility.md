# Reproduction and Claim Boundaries

Native arm64 `.venv-pytorch`: PyTorch 2.12.0, NumPy 2.4.6. CPU 4/inter-op 1,
single-process sampling, no resource probing. Existing model architecture,
27 fixed fits, 162,000 updates. No image encoder training in this matrix.

Registration SHA256:
`663cb62a13281fb3533e95d28b23814452560292e54cbcb428ca1e0dc6bbda7e`

Report SHA256:
`f5a0c5a56b3e4c2f131f214745bfe252e94ca74209ec98c1be7becb990d56954`

Run from repository root with the existing approved local caches:

```bash
.venv-pytorch/bin/python scripts/run_m3w_unit_frame_training.py --registration configs/m3w_unit_frame_training_v1.json
.venv-pytorch/bin/python scripts/run_m3w_unit_frame_training.py --registration configs/m3w_unit_frame_training_v1.json --replay
.venv-pytorch/bin/python scripts/verify_m3w_unit_frame_run.py --registration configs/m3w_unit_frame_training_v1.json
.venv-pytorch/bin/python scripts/analyze_m3w_unit_frame_training.py --registration configs/m3w_unit_frame_training_v1.json
.venv-pytorch/bin/python scripts/audit_m3w_unit_frame_oracle.py --registration configs/m3w_unit_frame_training_v1.json
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_observed_unit_frame.py tests/test_m3w_observed_unit_frame_v2.py tests/test_m3w_normalization_response.py tests/test_m3w_unit_frame_training.py tests/test_m3w_unit_frame_analysis.py tests/test_m3w_sdd_step_adapter.py
```

The trainer automatically resumes model, optimizer, sampler, Torch RNG and step
state after interruption. Completed invocation verifies artifacts and makes zero
updates; it is not a fresh training result. Replay recomputes predictions, not
training. A deliberate fresh replicate needs a separately registered output
identity; never delete the existing run to manufacture fresh provenance.

Observed execution: pilot PID 96284, 100 updates; full trainer PID 96476, 161,900
additional updates; replay PID 97023; resume subprocess PID 97064. All terminated
successfully. Checks establish 27 exact prediction replays and 82 immutable hashes
unchanged on completed resume. Main/source sample histograms and final main RNG
match each prior paired control; synthetic phase-boundary resume is exact.

No sealed labels opened, no held-score selection, no test threshold tuning.
Main/source input receipts and code dependencies are hash-bound. Raw arrays,
history/image caches and checkpoints are private local artifacts, not GitHub
content. Git contains code/config/reports/aggregate metrics only. Full legacy
test suite not rerun; 24 relevant tests pass.

The primary is equal-site past-normalized ADE, not raw-frame t+50. Site-bootstrap
intervals are descriptive over three exposed sites; there is no independent
confirmation, calibrated safety proof, model promotion or paper-ready claim.
