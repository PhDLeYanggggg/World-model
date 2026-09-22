# Execution Record

## Registered Inputs and Runtime

Registration and implementation were committed and pushed as `562b41d1` before
the first real fit. GitHub main was first checked at `a6489df5`. The preflight
verified 1,159 source bindings and the twelve source/seed views. The unchanged
unrelated staged-work fingerprint is
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

Native arm64 `.venv-pytorch/bin/python`; CPU computation threads 4, interop 1,
DataLoader workers 0. Disk check showed about 53 GiB free. No GPU or remote job
was started. Existing SSH configuration has no CREATE alias; remote assets and
current queue are not verified, not declared absent. This small head experiment
is suitable locally and does not depend on resolving the pending CREATE access.

## Real Pilot and Resume

The `coupa_seed17/terminal_repeat` pilot (PID 15618) completed 100 real updates,
exited 0 and saved its checkpoint. Recorded fitting time was 0.13046 seconds,
excluding preflight, hashing and data preparation. This is a small risk network
over existing forecast features, not full neural trajectory training.

The full matrix was launched with `--resume`, PID 15674. A live process sample
at elapsed 41 seconds showed 126.8% CPU and RSS 3,427,616 KiB (about 3.27 GiB).
The first head had reached 12,000 updates. A single resource sample is not peak
usage or a guarantee for the remaining matrix. Checkpoints save every 500 steps;
heartbeat events every 100 updates remain in the ignored private run directory.

At this record's first creation, full fitting/readout were still running/not_run.
Completion, replay and scientific results must be read from the later receipts,
not inferred from this pilot or registration. No threshold or role was changed.

## Pre-Readout Verification Erratum

Static checking during training found a verifier wiring error: the final
`check_contrast` call in the frozen `verify_m3w_prefix_cost.py` passes summary
dictionaries instead of their `ADE/by_scene` dictionaries. The model, objectives,
decisions, metrics and source bindings are unaffected. The frozen verifier is
retained unchanged so in-progress checkpoint identities remain valid. Use
`scripts/verify_m3w_prefix_cost_readout.py`, a versioned replacement with the
correct argument shape and its own code hash in the receipt. This is not a new
metric or an outcome-driven comparison. The mistake was found before readout;
no results were inspected to choose the fix.

The replacement also explicitly checks full-grid missing-label bounds and the
conjunctive gates. Two targeted tests exercise nested summary inputs and reject
an altered interval. The model and the frozen primary evaluator were not edited.

## Completed Training and Readout

The full matrix (PID 15674) finished with exit 0: 24 heads, 288,000 updates,
73,728,000 draws, unknown draws 0. Recorded fitting time totals 403.41033 seconds.
Each head has 48,792 parameters. The first pilot is included, not counted twice.
Evaluation (PID 16845), checkpoint replay (PID 16915), and the versioned separate
verifier all exited 0. The scientific primary gate failed; successful execution
does not imply model success. The result is in `conclusions.md`.

```sh
.venv-pytorch/bin/python scripts/run_m3w_prefix_cost.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_prefix_cost.py --view coupa_seed17 --arm terminal_repeat --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_prefix_cost.py --resume
.venv-pytorch/bin/python scripts/run_m3w_prefix_cost.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_prefix_cost.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_prefix_cost_readout.py
.venv-pytorch/bin/python scripts/audit_m3w_prefix_cost_veto.py
```

Twenty-six head/target/log-loss tests passed, plus two versioned-verifier tests
and four fixed-veto diagnosis tests. These are scoped checks, not the full legacy
suite. All model files, row-level decisions and run events stay in the ignored
private `data/stage_cvpr2027_experiments/prefix_cost_v1/` directory. Public JSON
contains aggregate results and source bindings, not input rows or model weights.

Analysis SHA256:
`732d6726d011a0f46101a0d8424a9b10baefbfa3c6c77f790a078cf263f561c4`.
The veto audit ran twice, both exit 0, with exactly identical immutable output:
`ec468e8b8bd329f5c4efbda303526288046db4326e9828b1a9e4339d8e3cf89f`.
All required training, readout and audit sessions have terminated successfully.
Checkpoint replay, separate formulas and diagnostic checks share data/preprocessing
and the same implementing agent. They are not independent research confirmation.
