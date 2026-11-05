# Reproduction

Run from the World checkout using native arm64 `.venv-pytorch/bin/python`.
Fresh execution phases: register, pilot, train, evaluate, verify. Each completed phase is immutable.
Resume an interrupted fitting phase with `--resume`; do not restart completed training.
Checkpoint every100updates; heartbeat file under the local private study directory.
The registration binds code/config/protocol. The training freeze binds every model receipt.
Reports require a verified analysis hash. The report command does not fit or choose a policy.
Run `scripts/report_m3w_moving_proposal_risk.py` after successful verification.
Public aggregates and code are sufficient to inspect claims, not to train without separately licensed data.
Reserved H/DUT/DroneCrowd remain closed. Original failures and undefined controls remain.
