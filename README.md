# M3W

M3W is my research project on real-world multimodal, multi-agent world modeling.

The question I am working on is simple to state and difficult to make honest:

> Given a scene and the recent motion of all nearby agents, can a model predict what happens next without looking into the future, and without making the easy cases worse?

I am interested in the part of world modeling where the scene, the agents, and the failure modes all matter at the same time. The repo is therefore not just model code. It also contains dataset conversion, leakage checks, causal baselines, external transfer tests, negative results, and the audit trail I use to keep the claims from drifting.

## Current State

The strongest version right now is a protected 2.5D multi-agent world-state model. It is not a true 3D world model, and it is not a foundation model.

The deployable part is a conservative policy sitting on top of strong causal motion baselines. The learned components estimate when a baseline is likely to fail, how much a switch might gain, and how much harm that switch might cause. If the model is not confident, it falls back to the safe baseline.

That is intentional. For this project, improving hard cases is not enough if the model breaks easy cases.

## What Works

The most reliable progress so far has come from protected selection rather than unrestricted neural rollout.

- On SDD, the current cost-aware selector gives a useful pixel-space raw-frame improvement while preserving easy cases.
- On external top-down pedestrian data, the protected selector repaired the `t+50` transfer failure and produced positive dataset-local raw-frame gains.
- Later protected policies added stronger waypoint and source/domain handling, but they still depend on conservative fallback.
- Neural latent-state work is now being tested behind that same safety floor instead of replacing it outright.

The short version: M3W has promising protected world-state behavior, but the protection is still part of the model, not a temporary detail.

## What Did Not Work

Several routes were useful because they failed clearly.

- Hard one-label baseline classification switched too aggressively.
- JEPA-style representation learning avoided collapse, but did not yet give reliable downstream lift.
- Zero-shot SDD-to-external transfer failed before coordinate, horizon, and goal-context repair.
- Latent distribution alignment reduced domain distance without necessarily improving prediction.
- Ordinary residual correction was not safe enough to deploy.
- Ungated Transformer and Hybrid dynamics did not beat the protected floor.

Those failures shape the current direction: learn richer dynamics, but deploy them only when they beat the protected selector under no-leakage evaluation.

## Claim Boundary

I keep these boundaries explicit:

- SDD results are pixel-space benchmark results.
- External top-down results are dataset-local unless timing and geometry are source-verified.
- `t+50` and `t+100` are raw-frame horizons, not seconds-level claims.
- Inferred scene, goal, and visual-prior labels are not human gold annotations.
- Stage5C latent generative execution is not enabled.
- SMC is not enabled.

This is still a 2.5D / pseudo-3D trajectory world-state scaffold moving toward stronger multimodal world modeling. I do not want the README to sell more than the experiments support.

## How To Read This Repo

The root README is only the public introduction. The detailed record lives elsewhere:

| Path | Purpose |
| --- | --- |
| `README_RESULTS.md` | experiment ledger and evidence notes |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of routes tried, failures, successes, and current quality |
| `outputs/m3w_neural_v1/` | model cards, data cards, and M3W reports |
| `outputs/stage42_long_research/` | long-run audits, ablations, gates, and source/domain reports |
| `outputs/stage43_latent_state/` | protected latent-state experiments |
| `research_state.json` | machine-readable current state |

Large datasets, caches, checkpoints, videos, raw third-party data, and local virtual environments are intentionally not committed.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training path is kept conservative:

- `num_workers = 0`;
- checkpoint and heartbeat support for long runs;
- resume support where training is long;
- CPU/MPS-safe execution;
- no x86_64 Conda + Intel OpenMP training path.

The basic verification command is:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Next Direction

The next useful step is not a bigger claim. It is better evidence.

I am working toward:

- broader external top-down validation;
- source-specific timing and geometry audits;
- safer raw-frame long-horizon transfer;
- neural dynamics that beat the protected selector instead of copying it;
- clearer ablations for scene, goal, interaction, and latent-state contributions.

M3W is promising, unfinished, and deliberately conservative. The standard I am using is simple: the result has to survive the checks that would normally make it easy to fool myself.
