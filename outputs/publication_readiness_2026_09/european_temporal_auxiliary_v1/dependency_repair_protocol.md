# Allocated Dependency Repair and Pilot Retry

At 15:57:37 UTC on October 5, pilot 37795593 was verified FAILED, exit 1:0,
after 96 seconds. The login-shell repair worked, but the original portable entry
point imports the data-audit chain, which requires pandas. It failed before the
optimizer, heartbeat or any checkpoint. This is not model failure or success.

A static audit of the registered import closure and a read-only inventory of
the owned M3W runtime found pandas, python-dateutil and six absent. All other
third-party imports are already provided. Exact additions match the functioning
local environment: pandas 3.0.3, python-dateutil 2.9.0.post0, six 1.17.0. The
existing Torch 2.12.0+cpu, NumPy 2.4.6, SciPy 1.17.1, sklearn 1.8.0, joblib and
every other installed package must remain unchanged. Only official PyPI binary
wheels are permitted; dependency resolution, upgrades and pip caching are off.

The repair runs on an allocated compute node, never on the login node. It checks
both M3W ownership markers, exact environment prefix, personal quota (10 GiB
reserve plus registered checkpoint capacity and a 512 MiB installation allowance),
a nonblocking runtime lock and absence of other active M3W jobs. A before/after
package inventory, pip consistency check and full original-entry-point import
must pass before the unchanged pilot command starts. The separate runtime
receipt is engineering evidence, not a model result.

One explicit retry archives job 37795593's script, intent and receipt and retains
its logs. It requires its exact terminal error and absence of any scientific
artifact or active M3W job. All current input manifests and numerical code
bindings are reverified. Submission intent is written before sbatch; uncertain
outcomes never permit an automatic second submission.

The registered 4 CPU / 16 GiB / two-hour pilot allocation now includes the
bounded dependency setup. Original TRAIN packets, seeds, optimizer, losses,
100-step pilot, 2,000-step formal budget and exact-resume test are unchanged.
No full training or validation readout starts until their existing gates pass.
No independent roles, deployment changes, Stage5C, SMC or other project changes.

```sh
.venv-pytorch/bin/python -m scripts.retry_m3w_temporal_dependency_pilot register
# Commit the lightweight registration and implementation before submitting.
.venv-pytorch/bin/python -m scripts.retry_m3w_temporal_dependency_pilot submit
.venv-pytorch/bin/python -m scripts.manage_m3w_temporal_auxiliary_create status
```

Keep the prior port and reader registrations unchanged: no original execution or
numerical source file changes. This additive runtime amendment has its own pinned
registration. The runtime receipt and real pilot outputs must be verified after
completion; an accepted job or import-only check does not satisfy the science gate.
