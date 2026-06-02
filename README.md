# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on real-world multi-agent world modeling.

The problem I care about is simple to say and hard to solve: given a top-down scene, a group of moving agents, their recent motion, nearby interactions, and a few safe physical baselines, can a model predict plausible futures without cheating by looking ahead?

I am not trying to make a flashy demo. I am trying to build a system whose successes and failures are both measurable: when it improves a hard case, when it protects an easy case, when it breaks under a new coordinate system, and when a neural component looks promising but does not actually help.

## Why I Am Building This

Most trajectory models are judged by average error. That is useful, but it hides the cases that matter most to me.

A model can look good on average while still doing the wrong thing in deployment:

- switching away from a safe baseline on easy samples;
- failing under a different dataset coordinate system;
- improving one horizon while damaging another;
- learning latent features that look structured but do not improve downstream decisions;
- using evaluation shortcuts that would not exist at inference time.

M3W is built around those failure modes. The project is as much about safe model selection and evidence quality as it is about raw prediction numbers.

## Current Best Result

The strongest deployable part of the project right now is a protected prediction policy, not an unconstrained end-to-end neural world model.

The policy starts from strong causal baselines, reads only past motion and legal scene or interaction features, estimates when switching is likely to help, and falls back when the sample looks easy or uncertain.

On the current external top-down pedestrian evaluation, the best frozen policy reports:

| Slice | Result |
| --- | ---: |
| Overall improvement | +13.48% |
| Raw-frame `t+50` improvement | +8.46% |
| Hard/failure improvement | +15.54% |
| Easy-case degradation | 0.041% |
| `t+50` bootstrap CI | [+7.69%, +9.15%] |

That is the current reliable deployment floor. The neural world-model track is active, but I do not promote a neural model unless it beats this protected policy under the same safety rules.

## What The Model Uses

M3W works with top-down, dataset-local world-state data. The current system combines:

- recent agent history;
- velocity, acceleration, heading, curvature, and stop/go signals;
- neighbor density, nearest-neighbor distance, and interaction proxies;
- scene and goal prototypes where they can be built without test leakage;
- causal baseline rollouts;
- horizon, dataset, scene, and domain metadata;
- safety heads for failure, gain, harm, and fallback decisions.

The neural branch experiments with Transformer, JEPA-style representation learning, hybrid dynamics heads, waypoint prediction, and protected residual policies. So far, the protected selector and full-waypoint guarded policies are more reliable than unprotected neural dynamics.

## What This Repo Contains

This repository is the research trail behind M3W. I keep the successful runs, the failed routes, the ablations, the leakage audits, and the model cards together so the claims can be checked later.

Good entry points:

| Path | What it is for |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | The main results ledger with metrics, claim boundaries, and negative results. |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | A detailed Chinese write-up of what I tried, what failed, and what worked. |
| `outputs/m3w_neural_v1/` | Neural-model reports and model-card style summaries. |
| `outputs/stage42_long_research/` | Cross-domain, safety, replay, and paper-claim evidence. |
| `outputs/stage43_latent_state/` | Latent-state experiments, feature-family ablations, and reviewer-style evidence. |
| [`research_state.json`](research_state.json) | Machine-readable snapshot of the current research state. |

Large raw datasets, derived caches, checkpoints, videos, images, and local virtual environments are intentionally not committed.

## What I Am Not Claiming

This part matters.

M3W is not a true 3D world model yet. It is not a foundation world model. SDD results are pixel-space results, not metric predictions. External results are dataset-local unless calibration is verified. `t+50` and `t+100` are raw annotation-frame horizons, not seconds.

Self-audited or inferred scene labels are not human gold labels. Stage5C latent generative execution has not been enabled. SMC has not been enabled.

The current claim is narrower and, I think, more useful: M3W is a protected 2.5D multi-agent world-state modeling project with strong safety and leakage discipline, a reliable selector-style deployment floor, and an active neural dynamics track that still has to earn deployment.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Basic test command:

```bash
.venv-pytorch/bin/python -m pytest tests
```

Training and evaluation scripts are written around checkpointing, heartbeat logs, resume support, CPU/MPS-safe execution, and single-process dataloading.

## Next

The next research step is to make the neural world-model branch earn a real contribution instead of merely copying the protected selector.

The immediate priorities are:

1. keep the protected policy as the safety floor;
2. prove whether scene, goal, and interaction context add measurable lift;
3. repair weak long-horizon slices, especially raw-frame `t+100`;
4. promote neural dynamics only if they beat the protected policy under the same no-leakage and easy-preservation rules.

If a route fails, I keep the failure in the record. If a route works, the repo should make clear exactly what worked, where it worked, and what the claim still does not cover.
