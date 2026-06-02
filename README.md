# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling.

The question behind the project is simple:

> If I can see a scene, the agents in it, their recent motion, and their local interactions, can I predict what happens next more reliably than strong causal motion baselines?

I started this repo to answer that question carefully, not just to collect a nice-looking demo. The work here includes the models that improved results, the ones that failed, the leakage checks, the safety rules, and the notes that keep me honest about what the evidence does and does not prove.

## Current Best Result

The strongest deployable model in this repo is currently a protected selector-style policy. It does not blindly replace the causal baseline. It starts from strong physical baselines, studies past-only evidence, and switches only when the expected gain is large enough and the easy-case risk is low enough.

On the current external top-down pedestrian evaluation, the frozen policy gives:

| Slice | Result |
| --- | ---: |
| Overall improvement | +13.48% |
| Raw-frame `t+50` improvement | +8.46% |
| Hard/failure improvement | +15.54% |
| Easy-case degradation | 0.041% |
| `t+50` bootstrap CI | [+7.69%, +9.15%] |

This is the result I trust most today. The neural world-model branch is still active, but I only count it as deployable when it beats this protected policy under the same no-leakage and easy-preservation rules.

## What The System Looks At

The current M3W pipeline works with dataset-local top-down trajectories. It uses information that would be available at inference time:

- recent agent history;
- speed, acceleration, heading, curvature, and stop/go behavior;
- neighbor density and interaction signals;
- train-only scene or goal context when that context is legally available;
- causal baseline rollouts;
- dataset, scene, horizon, and domain metadata;
- risk heads for failure, gain, harm, and fallback decisions.

I also maintain a neural track with Transformer dynamics, JEPA-style representation learning, hybrid heads, waypoint prediction, and protected residual policies. So far, the most reliable gains come from guarded selection, causal history windows, full-waypoint structure, domain-aware routing, and safety floors. The neural pieces are useful research evidence, but they still have to earn deployment.

## What This Repo Is For

This repository is a research record. The most important rule in the project is that a result has to survive the boring checks: no future leakage, no test endpoint goals, no central-velocity shortcuts, no easy-case damage hidden inside aggregate gains, and no metric claims without calibration.

For a quick orientation:

| File or directory | What to read it for |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | Detailed results ledger and current evidence boundaries. |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | Chinese long-form summary of routes tried, failures, causes, and successes. |
| [`research_state.json`](research_state.json) | Machine-readable snapshot of the current project state. |
| `outputs/m3w_neural_v1/` | Neural world-model reports and model-card style summaries. |
| `outputs/stage42_long_research/` | Cross-domain safety, replay, full-waypoint, and paper-claim evidence. |
| `outputs/stage43_latent_state/` | Latent-state, graph/history/context, and reviewer-style validation reports. |

Large datasets, caches, checkpoints, videos, images, third-party data, and local virtual environments are intentionally kept out of git.

## What I Am Not Claiming

M3W is not a true 3D world model yet. It is not a foundation world model. SDD results are pixel-space unless calibration is verified. External results are dataset-local unless their geometry is verified. `t+50` and `t+100` are raw annotation-frame horizons, not seconds. Self-audited or inferred labels are not human gold labels.

Stage5C latent generative execution has not been enabled. SMC has not been enabled.

The current claim is narrower and stronger: this repo contains a protected 2.5D multi-agent world-state modeling system with strict leakage discipline, a reliable selector-style deployment floor, and an active neural dynamics track that is still being tested against that floor.

## Running Locally

On Apple Silicon, training should use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Basic test command:

```bash
.venv-pytorch/bin/python -m pytest tests
```

Training scripts are written around checkpointing, heartbeat logs, resume support, CPU/MPS-safe execution, and single-process dataloading.

## Next Step

The next research step is to make the neural branch contribute something the protected policy does not already provide.

That means:

1. keep the protected selector as the safety floor;
2. promote neural dynamics only if they improve overall, `t+50`, or hard/failure slices without damaging easy cases;
3. keep testing whether scene, goal, graph, and latent context add measurable lift;
4. keep raw-frame and dataset-local claims separate from metric or physical-world claims.

When a route fails, I leave it in the record. When a route works, I want the repo to show exactly where it works, why it is allowed, and what it still does not prove.
