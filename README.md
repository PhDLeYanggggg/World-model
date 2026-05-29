# M3W

M3W is my attempt to build a real-world multimodal, multi-agent world model from top-down trajectory data.

The question I care about is simple to say and hard to answer honestly:

> Given a scene and the recent motion of the agents inside it, can a model predict what happens next without using future information, and without making the easy cases worse?

Most of this repository is not glamorous model code. It is the work around the model: converting messy datasets, checking leakage, building strong causal baselines, testing external transfer, recording negative results, and refusing to promote a result just because it looks good on one slice.

## Where The Project Stands

M3W is not a true 3D world model. It is not a foundation model. The current system is best described as a guarded 2.5D multi-agent world-state model.

The strongest deployable pieces right now are conservative selector policies. They start from causal motion baselines, estimate when switching to another baseline or learned head is likely to help, and fall back when the switch looks unsafe. That has been more reliable than replacing the baseline with an unrestricted neural network.

The current claim boundary is strict:

- Stanford Drone Dataset results are pixel-space results.
- External top-down results are dataset-local unless timing and geometry are verified for that source.
- `t+50` and `t+100` are raw-frame horizons, not seconds-level claims.
- Inferred scene, goal, and visual-prior labels are not human gold labels.
- Stage5C latent generative rollout has not been executed.
- SMC is not enabled.

I am trying to make this a stronger agent-scene world model over time, but I would rather understate the result than sell a model the evidence does not support.

## What Is Working

The clearest progress has come from protected, cost-aware selection.

On SDD, the best current policy improves the pixel-space benchmark while keeping easy cases stable. On external top-down pedestrian data, the useful policies combine causal history, train-only goal prototypes, source-aware validation, and conservative fallback rules. Recent long-run experiments also improved protected full-waypoint and raw-frame long-horizon evidence, but those results stay inside the dataset-local boundary.

The main lesson so far is that strong baselines are part of the model, not an inconvenience. They give the learned pieces a safety floor.

## What Has Not Worked Yet

Some directions were worth trying and are still useful as evidence, but they are not current main claims:

- hard one-label baseline classification switched too often and hurt easy cases;
- JEPA-style representation learning avoided collapse but has not shown reliable downstream lift;
- zero-shot SDD-to-external transfer failed before coordinate, horizon, and history repair;
- latent distribution alignment reduced domain distance without reliably improving prediction;
- ordinary residual correction was not safe enough to deploy;
- ungated Transformer and Hybrid dynamics did not beat the protected floor;
- scene/goal and neighbor/interaction features are diagnostic signals, not yet independent headline contributions.

Those failures are not swept away. They explain why the current system is guarded.

## Repository Map

This README is the public overview. The detailed research record is kept separately:

| Path | Purpose |
| --- | --- |
| `README_RESULTS.md` | long experiment ledger and stage-by-stage evidence |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of the routes tried, failures, successes, and current quality |
| `outputs/m3w_neural_v1/` | current model cards, reports, and evidence files |
| `outputs/stage42_long_research/` | long-run audits, ablations, gates, and source/domain reports |
| `research_state.json` | machine-readable current state |

Large local data, caches, checkpoints, videos, images, and third-party raw files are intentionally not committed.

## Running Locally

On Apple Silicon, I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training and evaluation scripts avoid the old x86_64 Conda + Intel OpenMP path:

- `num_workers = 0`;
- CPU-safe and MPS-safe execution paths where applicable;
- checkpoints, heartbeat files, and resume support for long runs.

Typical verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Next Direction

The next useful progress is evidence, not bigger language:

- broader external top-down validation;
- safer long-horizon raw-frame transfer;
- stronger held-out source and held-out scene coverage;
- neural dynamics that beat the protected selector instead of copying it;
- metric or seconds-level reporting only after source-specific timing and geometry audits justify it.

That is the standard for M3W: promising, unfinished, and measured by what survives audit.
