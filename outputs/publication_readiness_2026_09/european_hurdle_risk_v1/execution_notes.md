# Execution and Reproduction Receipt

## Registration and Actual Execution

Registration commit: `7a1cd0d9`, pushed before new training. Config, runner,
model implementation, model tests, fitting-support diagnosis and upstream
producer identities are hash-bound. No post-readout refit or threshold search.

Native `.venv-pytorch/bin/python`: arm64, Python 3.11.1, Torch 2.12.0,
CPU threads 4, inter-op 1, DataLoader workers 0. No MPS/resource probing,
Rosetta or default x86 Conda path. Measured local memory and runtime justified
local execution; no HPC training was needed. Peak observed RSS was about 7.4GB.

| Phase | PID | Terminal result | UTC observation |
|---|---:|---|---|
| Preparation | 49360 | Exit 0 | Finished 09:55:32 |
| 100-update pilot | 49424 | Exit 0 | Finished 09:56:14 |
| Training, head replay, evaluation | 49551 | Exit 0 | 09:58:15-10:03:22 |
| Independent metric verification | 50033 | Exit 0 | 10:03:55-10:06:09 |
| Completion report / tests | Recorded locally | Exit 0 | Receipt 10:06:21 |

72 heads each received 2,000 updates: 144,000 in total, with the pilot included.
Each head has 22,979 parameters. Sum of measured head-fitting time is 114.05s;
this excludes preparation, inference, replay, bootstrap and reporting, and is
not whole-experiment wall time. Checkpoints every 200 steps include optimizer,
RNG and sampler state. Heartbeat/PID records and private checkpoints remain
locally in the ignored experiment directory. Unknown-label training draws = 0.

## Reproduction Scope

All 72 heads reproduce scores on the first 4,096 excluded-index rows each, not
random rows or full-bank inference. All 72 sampler sequences match their
controls. Separate verification recomputes all 144 full policy metrics and
matches all 72 old/geometric control views. All required phases terminated.

218 tests across 34 named files pass; names and hashes are in
[completion_checks.json](completion_checks.json). This is not `pytest tests`
over the entire historical repository. Unchanged legacy suites were not rerun.
The [zero-reference audit](zero_reference_audit.json) is an additional posthoc
raw-prediction check and matches the primary harm counts without fitting.

Key SHA-256 receipts:

```text
analysis 29bb7dc9a39bda42699094b7176e313934766e2202e1d3b74b19e6845fb1b331
summary  0562ff7b6b84c10e61ba0b986ed4274caa0ecfaa3cbfad05a5fd0965c89d9f28
reporter 5db0f69e5aa3a5f6e0c2c7ea6506a95eef7ef9259d442a6ecabd50a9afd15040
loss CSV 7862b0def8045e5f22e1fccb307b251217ac6f1874ccc6e8c5ef42fd9dbf5037
tests    367cfa9300bdbadfb4a4555fea7b562886d601ee2643172f9760b022ee950860
```

Both aggregate SVG figures were rendered and visually inspected. The loss
figure compares the common moment-MSE component; unequal total objectives
must not be compared as if they were the same loss or validation performance.

## CREATE and Git Boundaries

The authorized read-only queue query at 09:47:01 UTC returned exit 0 with no
rows matching the query. This does not prove old jobs finished or failed.
No remote job was submitted, modified or cancelled. M3W's remote asset directory
remains unverified / `not_run`; no other project's directory was repurposed.

Public artifacts contain code, aggregate metrics, SVGs and reports only.
No raw trajectories, private caches, checkpoint banks, third-party images or
runtime environment are included. Reproduction requires the separately held
data and upstream producer chain, not merely a fresh Git clone. Unrelated
staged work is excluded from this task's scoped commit.

## Interpretation

Training and full metric recomputation are `fresh_run`; frozen forecast and
utility/control assets are `cached_verified`. Independent confirmation and
remote training are `not_run`. Twelve opened development localities, three
seeds and conditional locality bootstrap are not independent final-test
evidence. Read the [operation guide](operation_zh.md) before resuming, and
verify PID termination rather than interpreting a stale heartbeat as activity.

No deployment promotion, Stage5C or SMC. Image pixels and raw steps, not metric,
seconds, human-gold labels, physical safety, true 3D or foundation modeling.
