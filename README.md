# M3W

M3W is my research project on real-world multi-agent world modeling. I am trying to answer a practical question:

> If we know the scene and the recent motion of all visible agents, can we predict what happens next without peeking into the future and without breaking the easy cases?

The work is deliberately grounded in real top-down pedestrian and drone-style trajectory data. A lot of the project is not glamorous model-building. It is data conversion, leakage checks, strong causal baselines, negative results, and careful evaluation. That is the point: I want the result to be useful, not just impressive-looking.

## Current Status

The current system is a protected 2.5D multi-agent world-state model. It is a research candidate, not a finished foundation model.

What I can honestly claim right now:

- SDD results are pixel-space results.
- External trajectory results are dataset-local unless geometry has been verified.
- `t+50` and `t+100` mean raw-frame horizons, not seconds.
- The best deployed policies are guarded by causal fallback baselines.
- The repo includes failed routes because they explain why the current system is conservative.

What I do not claim:

- not true 3D;
- not metric-calibrated everywhere;
- not seconds-level unless a dataset audit proves it;
- not a large-scale foundation world model;
- not latent generative rollout;
- no SMC deployment.

## What Works

The strongest pieces so far are protected policies that start from strong causal motion baselines and switch only when the model has evidence that switching is safe.

The most useful results are:

- a cost-aware SDD selector that improves the pixel-space benchmark while preserving easy cases;
- an external `t+50` repair that turned failed transfer into a deployable raw-frame selector candidate;
- source-aware and history-aware policies with bootstrap and replay checks;
- no-leakage audits for future endpoints, central velocity, and test-built goals;
- a growing set of long-horizon and full-waypoint evidence under fallback protection.

In short: M3W is currently useful as a guarded trajectory and world-state research system. The longer-term goal is a stronger multimodal agent-scene world model.

## What Did Not Work

Some important ideas did not become main claims:

- hard one-label baseline selection switched too aggressively and damaged easy cases;
- JEPA representations did not collapse, but they have not yet produced reliable downstream lift;
- zero-shot SDD-to-external transfer failed before domain and horizon repair;
- latent distribution alignment reduced feature distance without reliably improving prediction;
- ordinary residual correction was not safe enough to deploy;
- ungated Transformer and Hybrid dynamics did not beat the protected floor.

I keep these failures in the repo because they are part of the research, not because I want the project to look bigger than it is.

## Repository Map

| Path | What it is for |
| --- | --- |
| `src/` | data processing, models, evaluators, gates, report builders |
| `configs/` | training and evaluation configs |
| `tests/` | regression tests and evidence checks |
| `outputs/m3w_neural_v1/` | current M3W-Neural summaries and model cards |
| `outputs/stage42_long_research/` | long-run reports, ablations, guards, and evidence tables |
| `README_RESULTS.md` | detailed experiment ledger |
| `research_state.json` | machine-readable project state |

Large data, local caches, checkpoints, third-party raw files, videos, and image assets are intentionally not committed.

## Local Runtime

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The default training/evaluation setup avoids the old x86_64 Conda + Intel OpenMP path:

- `num_workers = 0`;
- CPU-safe and MPS-safe paths where applicable;
- checkpoint, heartbeat, and resume support for long runs;
- no future endpoint input;
- no central velocity as official input;
- no test endpoints for goal construction.

Typical verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Direction

The next useful progress is not making the README louder. It is making the evidence harder to break:

- broader external top-down validation;
- stronger all-agent waypoint evidence;
- clearer source, horizon, and scene breakdowns;
- neural dynamics only where they beat the protected floor;
- metric and time calibration only when the dataset evidence supports it.

M3W is the path toward a real multimodal agent-scene world model. This repository shows the path as it actually is: promising, guarded, and still unfinished.
