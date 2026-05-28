# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on real-world multi-agent world modeling.

The question I care about is:

> Given a real top-down scene and the recent motion of every visible agent, can a model predict what happens next without peeking into the future, without leaning on weak shortcuts, and without making the easy cases worse?

That sounds simple, but most of the work has been in making the setup honest. The project tracks not only the models that worked, but also the baselines, failed ideas, leakage checks, and negative results that shaped the current direction.

## Where The Project Stands

The current system is a protected 2.5D multi-agent world-state model. It is not a true 3D world model, not a foundation model, and not a metric or seconds-calibrated predictor.

The current claims are intentionally narrow:

- SDD experiments are pixel-space.
- External pedestrian experiments are dataset-local unless geometry is verified.
- `t+50` and `t+100` are raw-frame horizons, not seconds.
- self-audited and visual-prior labels are not human-gold labels.
- latent generative execution has not been run.
- SMC is not enabled.

I would rather keep the project useful and defensible than make it sound more finished than it is.

## What Works So Far

The strongest result so far is not an unconstrained neural rollout. It is a protected policy that starts from strong causal motion baselines and only switches when the model has enough evidence that switching is safer and better.

The current best pieces are:

- a strong SDD pixel-space selector baseline;
- an external `t+50` repair that turned failed transfer into a deployable raw-frame selector candidate;
- source-aware, history-aware, and waypoint-aware policies that improve long-horizon behavior under a fallback floor;
- no-leakage checks that keep future endpoints, central velocity, and test-built goals out of inference;
- bootstrap and replay reports for the results that look promising.

In plain terms: M3W is useful today as a guarded trajectory/world-state research system. It is not yet the final world model I want it to become.

## What Did Not Work

Several routes were worth trying but did not become main claims:

- hard one-label baseline selection switched too often and hurt easy samples;
- JEPA representations did not collapse, but have not yet produced reliable downstream lift;
- zero-shot SDD-to-external transfer failed before domain and horizon repair;
- latent distribution alignment reduced distance without reliably improving prediction;
- ordinary residual correction was not safe enough to deploy;
- ungated Transformer and Hybrid dynamics did not beat the protected floor.

These negative results are part of the project. They are the reason the current model is conservative.

## How To Read This Repository

The root README is only the front door. The detailed research record lives in the reports.

| Path | Purpose |
| --- | --- |
| `src/` | data processing, model code, evaluators, gates, report builders |
| `configs/` | training and evaluation configs |
| `tests/` | regression tests and evidence checks |
| `outputs/m3w_neural_v1/` | current M3W-Neural summaries and model cards |
| `outputs/stage42_long_research/` | long-run reports, ablations, guards, and evidence tables |
| `README_RESULTS.md` | detailed experiment ledger |
| `research_state.json` | machine-readable project state |

Large local data, caches, checkpoints, third-party raw files, videos, and image assets are intentionally not committed.

## Reproducing The Local Setup

On my Apple Silicon machine I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Important runtime choices:

- `num_workers = 0`
- CPU-safe and MPS-safe paths where applicable
- checkpoint, heartbeat, and resume support for long runs
- no future endpoint input
- no central velocity as official input
- no test endpoints for goal construction

Typical verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Current Direction

The next version of the project is about making the result harder to break:

- broader external top-down validation;
- stronger all-agent waypoint evidence;
- clearer source, horizon, and scene breakdowns;
- neural dynamics only where they beat the protected floor;
- metric and time calibration only when the dataset evidence supports it.

My longer-term goal is still a real multimodal agent-scene world model. The current repository is the honest research path toward that goal, not a polished claim that the goal is already solved.
