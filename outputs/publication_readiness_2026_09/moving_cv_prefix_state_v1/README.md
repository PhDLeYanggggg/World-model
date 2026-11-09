# Does causal recording state improve intervention?

I made each query decision aware of predicted gain, risk, and
reference support accumulated from earlier forecasts. The state
starts at zero for every recording. It does not use earlier realized
errors, later query histories, or the final recording length.
The main comparison follows a fixed causal portfolio for the
prefix, so the options and counts are identical. A second version
updates the same state with its own past decisions.

This is related to [online allocation under an unknown horizon](https://arxiv.org/abs/2206.13606),
but the risk balances here are predictions with uncertain
denominators, not known resource costs. I do not claim regret or
risk guarantees. The risk target measures a candidate's change in
positive excess relative to keeping CV at the current query. Both
models learn the same targets and differ only in access to the
prefix features. Training uses outcomes from fitting sources only.

## Development results

18 arm64 PyTorch fits, 25,200 updates, three source rotations and
three seeds. Held development has 24 recordings in nine source
groups, or 72 recording/seed views. The first five policies below
have identical intervention counts in every query.

| Policy | Edits | Original risk passes / 72 | Undefined easy / 72 | Worst known signed easy change |
|---|---:|---:|---:|---:|
| Frozen portfolio | 45,042 | 4 | 14 | +10.21% |
| Matched independent | 45,042 | 3 | 17 | +25.85% |
| Masked state | 45,042 | 3 | 14 | +38.75% |
| Causal state | 45,042 | 3 | 14 | +42.88% |
| Closed loop | 45,042 | 3 | 14 | +43.51% |
| Masked state, free counts | 52,477 | 3 | 22 | +125.82% |
| Causal state, free counts | 35,770 | 3 | 26 | +125.83% |
| Closed loop, free counts | 40,621 | 2 | 32 | +144.94% |

| Contrast | Observed-only pp | Completion-lower pp (95% CI) |
|---|---:|---:|
| Causal vs masked state, matched | -0.008 | -0.091 [-0.170, -0.034] |
| Causal state vs independent, matched | -0.371 | -1.191 [-1.706, -0.740] |
| Closed loop vs shadow state, matched | +0.009 | -0.003 [-0.031, +0.021] |
| Causal vs masked state, free counts | +0.029 | -0.474 [-0.857, -0.196] |
| Closed loop vs shadow state, free counts | +0.042 | -0.472 [-1.287, -0.005] |
| Direct arithmetic, state vs local, matched | +0.008 | -0.064 [-0.113, -0.020] |
| Matched state vs CV | +0.301 | -0.331 [-0.866, +0.127] |
| Matched state vs OLS4 | -0.300 | -54.963 [-96.537, -22.793] |

The direct arithmetic control requires no new model. It changes
2,558 row actions and passes 3/72 with or without state.
The learned state changes 7,939 row actions versus the masked
control. Conservative pair-distance gain is -0.083 local pixels
[-0.145, -0.034] for that matched comparison. Intervals use 3,000
paired source resamples after averaging seeds. Calibration qualifies
0/9; no learned policy is deployed.

## What failed

On matched-count held options, adding state reduces normalized
dual-gain MAE by 0.0055 [0.0014, 0.0097]. It does not establish
lower positive-harm MAE and worsens signed-easy MAE: masked minus
state is -0.0007 [-0.0014, -0.0001]. The masked/state arms pass
16/15 views on observed data alone, but only 3/3 under the original
missingness treatment. Each exceeds the positive easy-harm limit
in 55 views and has undefined easy support in 14.

The state is informative about gain without giving a better risk
decision. Using the same state directly also loses, so replacing
only the final neural scorer is unlikely to fix this particular
budget estimate. Next, test separately estimated positive easy
harm, missing-step debit, and reference support at matched coverage,
before carrying their estimates into a recording controller.

## Reproduction

```sh
.venv-pytorch/bin/python scripts/train_m3w_cv_prefix_state.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_cv_prefix_state.py
.venv-pytorch/bin/python scripts/eval_m3w_cv_prefix_arithmetic.py
```

Fits, scoring, arithmetic and prediction audit are `fresh_run`.
The shape model and original analytic portfolio are `cached_verified`.
Replay produces the frozen readout with zero optimizer updates.
Omit `--replay` to train or resume. The 113-second fresh run used
four Torch threads and no DataLoader workers; checkpoint and
heartbeat are local. No new CREATE job was needed.

Native eight observed / twelve forecast steps in dataset-local
pixels. No seconds, metric geometry, foundation or physical safety
claim. The 15/30 DroneCrowd roles remain fixed and confirmation
is `not_run`. No deployment, Stage5C, SMC or submission.

[Protocol](protocol.md) | [Learned readout](readout.json) |
[Prediction audit](prediction_audit.json) |
[Arithmetic control](arithmetic_readout.json) |
[Working supplement](../evidence_manuscript_v144/README.md).
