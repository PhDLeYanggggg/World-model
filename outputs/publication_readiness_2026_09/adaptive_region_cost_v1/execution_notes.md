# Adaptive Region Execution Record

## Scope

Registration6e4bfb88 was pushed before fitting. Current config, code, parent
evidence, checkpoints and upstream data are bound by1108source hashes. Parent
hashes and original data roles remain unchanged. Runtime: nativearm64 Torch2.12.0,
CPU4/inter-op1,workers0; no NumPy fallback, new CREATE job or remote-queue claim.
Available local disk was54GiB before execution. CPU is appropriate for these
small heads; no hours-long or large-world-model runtime claim is made.

## Commands

```bash
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_adaptive_region_cost.py tests/test_m3w_conditional_cost_head.py tests/test_m3w_conditional_cost_protocol.py tests/test_m3w_conditional_cost_audit.py tests/test_m3w_tempered_cost_head.py tests/test_m3w_tempered_cost_protocol.py tests/test_m3w_bounded_cost_head.py
.venv-pytorch/bin/python scripts/run_m3w_adaptive_region_cost.py --audit-only
.venv-pytorch/bin/python scripts/run_m3w_adaptive_region_cost.py --view coupa_seed17 --stop-at 100
.venv-pytorch/bin/python scripts/run_m3w_adaptive_region_cost.py --resume
.venv-pytorch/bin/python scripts/run_m3w_adaptive_region_cost.py --evaluate
.venv-pytorch/bin/python scripts/run_m3w_adaptive_region_cost.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_adaptive_region_cost.py
```

46 scoped tests passed in2.44s, including resumed equality at both refresh
boundaries and nonboundaries, unchanged sample draws, first-block equality to
the frozen trainer, and no label parameter in the refresh function. These are
synthetic/engineering checks, not independent research replication. Unchanged
tests are reused, not counted as new evidence. The nonhermetic full legacy test
suite was not rerun or reported as green.

An initial pilot command used the invalid namehold_coupa_seed17; the guard
rejected it before training. The corrected coupa_seed17pilot completed100steps
and resumed in the matrix, not an extra fit. A hash-display utility also hit a
locale error; settingLC_ALL=C displayed the digest successfully. Neither event
was an OpenMP hang or a changed experiment.

Checkpoints include model, optimizer, sampler/RNG, weights, every refresh model
snapshot and selection vector. Interrupted execution must resume the last atomic
checkpoint. The runner lock prevents a simultaneous duplicate. Inputs/targets
and final decision archives retain the approved offline-annotation scope.

Fresh fitting completed12heads,144000updates,36864000draws,276refreshes. Recorded
head-fit177.97723s includes36.91395s refresh inference. Loading/hashing, upstream
forecast fitting, metric calculation and verification are not part of that timer.
All inference decisions were frozen before evaluation target reads. Checkpoint
endpoint replay reproduced all527268scores and the saved analysis exactly.

Separate-formula verification completed: 276 refreshes, 12 initial weight vectors,
36 policy choices, 288 scene reductions and 1,581,804 supervision rows agree.
The receipt is independent_verification.json. It shares preprocessing/model forward with the trainer; its label,
refresh-selection, normalization, policy and metric arithmetic are separate.
It is same-agent verification, not an independent research-team replication.
No threshold or weight sweep was run after readout. Analysis and registration
are immutable; conclusions may summarize them but must not overwrite them.

Raw data, histories, decision arrays and all Torch checkpoints remain under
ignoreddata/stage_cvpr2027_experiments/adaptive_region_cost_v1. Only code/configs,
reports, scalar metrics and hash receipts are for Git. Stage5C/SMCoff; no deployment.
