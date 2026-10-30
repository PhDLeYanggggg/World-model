# Local reproduction and evidence roles

## Frozen experiment

Use native arm64 `.venv-pytorch/bin/python`, CPU4/interop1 and workers0. The
runner rejects Intel/Rosetta before importing Torch. Check the actual PID,
heartbeat and log before resuming; do not duplicate a healthy process. The
scientific code, config, source roles and registration must remain unchanged.
The 12h limit is a healthy-run allowance, not evidence that a timeout is success.

Original workflow, in a workspace with the registered local data/checkpoints:

```sh
.venv-pytorch/bin/python scripts/run_m3w_selected_query_critic.py register
# Commit code/config/protocol/registration before training.
.venv-pytorch/bin/python scripts/run_m3w_selected_query_critic.py pilot
.venv-pytorch/bin/python scripts/run_m3w_selected_query_critic.py train
.venv-pytorch/bin/python scripts/report_m3w_selected_query_critic.py training
.venv-pytorch/bin/python scripts/run_m3w_selected_query_critic.py eval
.venv-pytorch/bin/python scripts/run_m3w_selected_query_critic.py verify
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_selected_query_critic_v1
.venv-pytorch/bin/python scripts/report_m3w_selected_query_critic.py complete
```

Completed artifacts refuse replacement. If training or evaluation genuinely
stops before completion, inspect its original heartbeat/log/PID, then use the
same phase with `--resume`. Checkpoints bind features, labels, model settings,
initialization, source/query identities, random generators and draw hashes.
Resume is not permission to alter scientific settings. A fresh study requires
a new identity and registration, not deletion of this evidence.

Private assets live under
`data/stage_cvpr2027_experiments/european_selected_query_critic_v1/`.
Checkpoints, histories, raw source packets and per-row decision arrays are not
public Git artifacts. Published aggregate metrics are not model weights.

## Verify public aggregate evidence

```sh
.venv-pytorch/bin/python scripts/bundle_m3w_metrics.py european_selected_query_critic_v1 --verify
.venv-pytorch/bin/python scripts/report_m3w_selected_query_critic.py complete
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_selected_query_critic_artifacts.py
```

The archive contains the exact completed group JSON bytes. Canonical reduction
uses completed-roster order, not archive lexical order. Report reconstruction
uses these fixed aggregates and public training receipts; it does not retrain
or perform another policy evaluation. Exporting training traces again requires
the private original checkpoints. This is not a verified data-free cold-start
reproduction package or independent scientific confirmation.

## One-factor successor

The query-unit repair has a new name, registration, private process lock,
checkpoints and outputs. Its 36 controls are the original critics, not rerun
and called newly trained. The successor pilot freshly reproduces both old
100-step controls, then verifies new interrupted recovery. Its new full policy
readout is admitted only after this original complete replay succeeds. Both
full cohorts and the original risk requirement remain intact.

```sh
.venv-pytorch/bin/python scripts/run_m3w_query_unit_critic.py pilot
.venv-pytorch/bin/python scripts/run_m3w_query_unit_critic.py train
.venv-pytorch/bin/python scripts/report_m3w_query_unit_critic.py training
.venv-pytorch/bin/python scripts/run_m3w_query_unit_critic.py eval
.venv-pytorch/bin/python scripts/run_m3w_query_unit_critic.py verify
.venv-pytorch/bin/python scripts/report_m3w_query_unit_critic.py complete
```

Register/commit its already frozen definition before executing this sequence in
a genuinely fresh study. These commands describe the workflow, not permission
to overwrite the current completed pilot/training. No Stage5C, SMC, test-role
opening, metric/time conversion or deployment promotion is part of reproduction.
