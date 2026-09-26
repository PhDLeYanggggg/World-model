# Running and Resuming the Gradient Diagnostic

Use the native arm64 environment from the project root. Entry checks reject
Rosetta before importing Torch. Runtime is CPU4, interop1, DataLoader workers0.
No remote jobs are required. The fresh CREATE read-only receipt was obtained
without changing existing jobs; its hash is in registration_lock.json.

```sh
.venv-pytorch/bin/python scripts/run_m3w_european_aux_gradient.py --phase run
.venv-pytorch/bin/python scripts/report_m3w_european_aux_gradient.py
.venv-pytorch/bin/python scripts/plot_m3w_european_aux_gradient.py
.venv-pytorch/bin/python scripts/verify_m3w_european_aux_gradient.py
```

Registration was committed and pushed in 1a5db084 before the real-data pilot.
The runner verifies original inputs, labels, source bindings, model states
and optimizer receipts. It reuses complete per-view receipts on resume.
An interrupted current view is recomputed from immutable source checkpoints;
no partial diagnostic changes its source model. The file lock prevents a
duplicate runner. Parent full training is not repeated by these commands.

Commit the diagnostic_freeze.json before aggregate readout. Verification
recomputes every isolated update and requires byte-identical diagnostics,
report and figure. This is diagnostic replay, not a second independent fit.
The pilot completed 120 virtual updates over one view in 16.431732 seconds
including that view's data preparation, excluding ancestry preflight.

Before aggregate readout, the integer producer/controller display IDs needed
explicit string formatting. The public implementation amendment records the
old/new hashes of that one-line report fix and the runner's strict amendment
validation. The original registration is retained, not rewritten. The
running experiment had loaded the registered original runner; no numerical
diagnostic, source data, model, loss, probe or allocation rule changed. Replay
uses the amended validation and must reproduce every diagnostic byte-for-byte.
The first aggregate save subsequently exposed a NumPy-boolean serialization
failure. Native-float conversion preserves the numeric value and produces
standard Python comparisons. The public amendment links its predecessor;
a complete synthetic aggregate test covers this failure. No scientific
threshold, metric, grouping, probe, model or real update was changed.

Private files, PID heartbeat, events and receipts live under
data/stage_cvpr2027_experiments/european_aux_gradient_v1/. Keep all row-level
diagnostics private. No data, images, histories, latent cache or checkpoints
are committed. Preserve 10GiB free disk and all unrelated staged work.

Fitting probes are not validation or final tests. No deployment, calibration,
threshold, final-test role, Stage5C or SMC is changed by this experiment.
