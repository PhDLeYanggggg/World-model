# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my long-running research project on real-world top-down multi-agent world modeling.

The question I am working on is:

> Given a scene, a set of moving agents, their recent histories, local interactions, and only information that would be available at inference time, can a model predict future motion more reliably than strong causal baselines?

This repository is not a polished product demo. It is the research record behind that question: the models that worked, the ones that failed, the leakage audits, the safety gates, and the evidence I use to decide whether a result is actually deployable.

## Where The Project Stands

The strongest deployable result right now is not an unconstrained neural world model. It is a protected policy that starts from strong causal baselines and only switches away from them when past-only evidence suggests the switch is safe.

On the current external top-down pedestrian evaluation, the frozen Stage37 policy is the best deployment floor:

| Evaluation slice | Result |
| --- | ---: |
| Overall improvement | +13.48% |
| Raw-frame `t+50` improvement | +8.46% |
| Hard/failure improvement | +15.54% |
| Easy-case degradation | 0.041% |
| `t+50` bootstrap CI | [+7.69%, +9.15%] |

That is the result I trust most today. Later neural dynamics experiments are kept in the repo, but I do not treat them as deployable unless they beat this floor under the same no-leakage and easy-preservation rules.

## What M3W Uses

The current system works on dataset-local top-down trajectory data. It uses:

- recent agent history;
- velocity, acceleration, heading, curvature, and stop/go signals;
- neighbor density and interaction proxies;
- train-only scene or goal context where it is legally available;
- causal baseline rollouts;
- horizon, dataset, scene, and domain metadata;
- failure, gain, harm, and fallback heads for safe selection.

The neural track explores Transformer dynamics, JEPA-style representation learning, hybrid heads, waypoint prediction, and protected residual policies. So far, the reliable gains come from guarded selection, causal history, full-waypoint structure, domain-aware routing, and safety floors. JEPA and unprotected neural dynamics remain research evidence, not final claims.

## How To Read This Repo

I keep the public README short on purpose. The detailed evidence is split into ledgers and reports:

| File or directory | What it contains |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | Main results ledger, current claim boundaries, and stage-by-stage evidence. |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | Detailed Chinese summary of attempted routes, failures, causes, and successes. |
| [`research_state.json`](research_state.json) | Machine-readable snapshot of the current project state. |
| `outputs/m3w_neural_v1/` | Neural world-model reports and model-card style summaries. |
| `outputs/stage42_long_research/` | Cross-domain safety, replay, full-waypoint, and paper-claim evidence. |
| `outputs/stage43_latent_state/` | Latent-state, graph/history/context, and reviewer-style validation reports. |

Large datasets, caches, checkpoints, videos, images, third-party data, and local virtual environments are intentionally not committed.

## What I Am Not Claiming

These boundaries are important:

- M3W is not a true 3D world model yet.
- M3W is not a large-scale foundation world model.
- Current SDD results are pixel-space, not metric.
- External results are dataset-local unless calibration is verified.
- `t+50` and `t+100` are raw annotation-frame horizons, not seconds.
- Self-audited or inferred labels are not human gold labels.
- Stage5C latent generative execution has not been enabled.
- SMC has not been enabled.

The current claim is narrower: M3W is a protected 2.5D multi-agent world-state modeling project with strong leakage discipline, a reliable selector-style deployment floor, and an active neural dynamics track that still has to earn deployment.

## Running Locally

On Apple Silicon, training should use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Basic test command:

```bash
.venv-pytorch/bin/python -m pytest tests
```

Training scripts are designed around checkpointing, heartbeat logs, resume support, CPU/MPS-safe execution, and single-process dataloading.

## Next Research Step

The next step is to make the neural world-model branch contribute something the protected policy does not already provide.

In practical terms, that means:

1. keep Stage37 as the safety floor;
2. only promote neural dynamics if they improve all, `t+50`, or hard/failure slices without damaging easy cases;
3. keep testing whether scene, goal, graph, and latent context add measurable lift;
4. keep raw-frame and dataset-local claims separate from metric or physical-world claims.

If a route fails, it stays in the record. If a route works, the repo should make clear exactly where it works, why it is allowed, and what it still does not prove.
