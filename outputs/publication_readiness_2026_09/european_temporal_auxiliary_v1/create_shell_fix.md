# CREATE Login-Shell Repair

Result source: fresh scheduler/log observation and engineering repair; no model
result. On October 5 at 15:22:36 UTC, job 37790290 was verified FAILED, exit
127:0, elapsed one second. Its error is `line 14: module: command not found`.
No heartbeat, checkpoint or pilot result exists. The error occurs before Python.
This replaces the earlier unknown-state observation, not the historical record.

A read-only CREATE login-shell probe at 15:25:21 UTC loaded the exact registered
module and reported Python 3.11.6, exit zero. It did not run numerical training.
The pre-fix header regression failed as expected; 95 related tests now pass,
including exact failure archival, repeat safety and refusal for existing work.

The submission used `#!/bin/bash`; it must use `#!/bin/bash -l`, as the earlier
successful M3W CREATE runtime/training submissions do. Execution revision 4
changes that header, not the optimizer, inputs, model, sampler, budgets, threads,
CPU partition, memory, reserve, walltime, resume rule or scientific protocol.
No environment, credential, network or other project configuration is changed.

The repair utility requires the exact failed job and exit code, its matching
module error, empty stdout, zero checkpoints/heartbeat, no full submission and
no active temporal M3W job. It verifies input manifests and remote code hashes.
It then archives the failed submission receipt, intent and batch script before
allowing a separate explicit retry. Original logs remain in place. Unknown
submission outcomes or active jobs never authorize a retry. Repeated archival
requests verify the existing receipt rather than creating another submission.

Both port and reader registrations retain revision 3 and record new revision 4
bindings. Input manifests and numerical training code are unchanged. Full
training remains prohibited until the new real-data pilot verifies finite
outputs, exact interruption/resume and the registered time/memory requirements.
All 216 fixed-final fits must finish before validation readout. Independent
roles stay closed; deployment, Stage5C and SMC stay unchanged/off.

Reproduction after committing registration:

```sh
.venv-pytorch/bin/python -m scripts.repair_m3w_temporal_create_shell
.venv-pytorch/bin/python -m scripts.manage_m3w_temporal_auxiliary_create submit-pilot
.venv-pytorch/bin/python -m scripts.manage_m3w_temporal_auxiliary_create status
```

Do not repeat the submission command after an uncertain outcome. Inspect the
remote intent/receipt and scheduler first. This repair is not training success;
the new job's actual artifacts must be checked independently.
