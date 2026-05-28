# M3W

M3W is my research project on real-world multimodal, multi-agent world modeling.

The problem I am trying to solve is simple to state and hard to do honestly:

> From a real top-down scene and the recent motion of all visible agents, predict the next world state without using future information and without making easy cases worse.

I am not treating this as a demo project. The repository keeps the useful results, the failed routes, the safety checks, and the uncomfortable audit notes because those are what made the current system better.

## Current Status

M3W is currently a protected 2.5D multi-agent world-state model. It is not a true 3D world model, not a foundation model, and not a metric or seconds-calibrated system.

The current evidence should be read with these boundaries:

- SDD results are pixel-space.
- External top-down pedestrian results are dataset-local unless the source geometry is verified.
- t+50 and t+100 are raw-frame horizons, not seconds-level claims.
- self-audited and visual-prior labels are not human-gold labels.
- latent generative execution has not been run.
- SMC is not enabled.

I would rather keep the claim narrow and defensible than make the project sound more finished than it is.

## What Is Working

The strongest pattern so far is a protected policy, not an unconstrained neural replacement.

It starts with a strong causal baseline, then uses causal history, neighbor context, route and goal prototypes, source/domain context, and learned risk signals to decide whether switching is worth it. If the case looks easy, uncertain, or likely to be harmed, the model falls back.

That design has produced the most reliable evidence so far:

- an SDD pixel-space cost-aware selector that remains the SDD safety baseline;
- an external t+50 repair that turned failed transfer into a deployable raw-frame selector candidate;
- protected full-waypoint and group-consistency policies that move beyond endpoint-only behavior;
- reviewer-style replay packages and no-leakage checks that make the results easier to audit.

My current short description is:

> M3W is a protected raw-frame 2.5D multi-agent world-state candidate. It has useful evidence on the current benchmarks, but it is still bounded, safety-floored, and not yet a true 3D or foundation world model.

## What Did Not Hold Up

Some directions were useful scientifically but did not become main claims:

- hard one-hot baseline classification over-switched and damaged easy cases;
- JEPA did not collapse, but it has not yet given reliable downstream lift;
- zero-shot SDD-to-external transfer failed before domain and horizon repair;
- latent alignment reduced distribution distance without reliably improving prediction;
- ordinary residual correction was not safe enough to deploy;
- ungated Transformer and Hybrid dynamics did not beat the protected floor;
- scene, goal, and interaction context help inside guarded policies but are not yet standalone contributions.

I keep these results in the repo because they explain why the current system is conservative.

## Repository Map

The detailed experiment ledger is separate from this public overview.

| Path | What it is for |
| --- | --- |
| `src/` | data processing, models, evaluators, gates, report builders |
| `configs/` | training and evaluation configs |
| `tests/` | regression tests and evidence checks |
| `outputs/m3w_neural_v1/` | current M3W-Neural v1 summaries and model cards |
| `outputs/stage42_long_research/` | long-run evidence, gates, claim guards, ablations |
| `README_RESULTS.md` | internal experiment ledger |
| `research_state.json` | machine-readable project state |

Large local data, caches, checkpoints, third-party raw files, videos, and image assets are not committed.

## Reproducibility

On my Apple Silicon machine I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Important runtime choices:

- `num_workers = 0`
- CPU-safe and MPS-safe paths where applicable
- checkpoint / heartbeat / resume support for long runs
- no future endpoint input
- no central velocity as official input
- no test endpoints for goal construction

Typical verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Current Direction

The next step is to make the claim harder to break:

- broaden external top-down validation;
- strengthen all-agent waypoint evidence;
- keep source / horizon / scene breakdowns visible;
- improve neural dynamics only where it beats the protected floor;
- continue separating deployable results from diagnostic ones;
- attempt metric and time calibration only when the source evidence supports it.

Until those pieces are stronger, I describe M3W as a protected 2.5D world-state candidate, not a finished world model.
