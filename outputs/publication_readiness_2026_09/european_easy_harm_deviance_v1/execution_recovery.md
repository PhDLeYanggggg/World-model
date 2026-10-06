# Execution And Recovery Boundaries

This experiment changes a cost-head loss, not the frozen trajectory forecaster.
Its144 registered fits and2,000 updates per fit must complete before any new
development readout. The original2% selected positive easy-harm/reference
budget is not the same quantity as net easy ADE degradation.

## Current Receipts

- Pilot37813826: completed0:0;100 updates per arm, plus exact replays.
- Full array37814169: not complete. Task0 failed the exact historical control
  after2,000 updates. Its atomic checkpoint remains available.
- Tasks2/3 were held before starting. Task1 completed0:0 in5m17s; all36 final
  checkpoints and18 historical controls were verified at10:28 UTC.
- Same-node control diagnostic37814380 completed0:0 in32s. Original/new direct
  and pilot-resumed states match exactly; only the historical artifact differs.
- An explicit one-identity reference amendment preserves that failure and uses
  the verified original-trainer replay, without increasing floating tolerances.
- Replacement array37815216 resumes only shards0/2/3. Join37815222 must combine
  them with unchanged completed task37814169_1. The old held tasks2/3 and
  obsolete join37814170 were cancelled after replacement receipts existed.

## Safe Inspection

Use the native local arm64 environment to inspect the owned jobs:

```sh
.venv-pytorch/bin/python -m scripts.manage_m3w_easy_harm_deviance status --phase train
.venv-pytorch/bin/python -m scripts.manage_m3w_easy_harm_control_diagnostic status
```

The manager reads only owned M3W records. It does not use simulation or clinical
data, install an environment, cancel unrelated jobs or run numerical work on a
login node. A failed observation is not proof that training failed or stopped.
Do not duplicate a submission when an intent or scheduler job already exists.

The short diagnostic compares direct original/new quadratic training, resuming
the new pilot's original control, and the historical/failed checkpoints. It runs
on the failed task's node to distinguish a code difference from execution
portability. Its registration leaves the exact acceptance rule unchanged.

```sh
.venv-pytorch/bin/python -m scripts.manage_m3w_easy_harm_control_diagnostic collect
```

Collection requires successful terminal scheduler accounting. A completed
heartbeat alone is insufficient. The verified result and
[explicit amendment](control_replay_amendment.md) retain the unresolved low-level
historical floating-point cause. Old receipts/logs are archived, not overwritten
without provenance. Original checkpoints and the36 completed fits remain.
Floating tolerances are not increased.

## Data And Storage

Existing24 TRAIN packets stay read-only under the parent experiment. No new
validation or independent-role packet is transferred. Checkpoints remain in
owned CREATE storage; public Git records contain only code, config and light
metadata. Preserve10GiB storage reserve, the shared256MiB checkpoint cap and
2MiB atomic-write headroom. No artifact is deleted to force admission.

Runtime:4CPU threads, interop1, workers0. Each training shard is limited to12h
with100-update checkpoints. A real hang, resource failure or failed invariant
requires diagnosis, not a smaller scientific sample passed off as completion.

No independent confirmation, metric/seconds conversion, deployment promotion,
Stage5C execution or SMC is authorized by a successful execution check.
