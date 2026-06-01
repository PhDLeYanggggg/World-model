# M3W

**Real-World Multimodal Agent-Scene World Model**

M3W is my research project on real-world multi-agent world modeling from top-down trajectory data. The question I keep coming back to is simple:

> Given only past motion, nearby agents, scene cues, and causal physical baselines, can a model predict future multi-agent behavior more reliably without looking into the future?

This repository is not a polished demo. It is a research workspace: I use it to build stronger baselines, test ideas that look promising, keep the failures, and make sure every useful result survives leakage checks and harder slices.

## What The Project Is About

Most trajectory models can look good on an average metric while still failing in ways that matter:

- they switch away from a simple physical baseline on easy cases and make things worse;
- they work on one dataset but break when the coordinate system, horizon, or scene distribution changes;
- they learn a latent space that looks tidy but does not improve downstream prediction;
- they improve one hard slice while quietly damaging the safer part of the distribution.

M3W is built around a more conservative idea: learning should earn the right to override the strongest causal baseline. If the learned policy is uncertain, if the expected gain is too small, or if the sample looks easy enough for the baseline, the system falls back to the safer baseline.

That safety floor is not a cosmetic detail. It is the main reason the current best models are deployable candidates rather than just interesting experiments.

## Current Status

The strongest reliable line right now is a protected multi-agent prediction policy. It combines:

- causal motion baselines;
- cost-aware selector logic;
- past-only history windows;
- neighbor and density features;
- scene-agnostic goal prototypes;
- validation-selected safety thresholds;
- strict fallback when switching is risky.

The project has produced positive evidence on SDD and external top-down pedestrian data under raw-frame, dataset-local evaluation. The most important practical lesson so far is that safe switching beats naive residual prediction, hard baseline classification, and unprotected neural dynamics.

The neural world-model track is still active, but I do not treat a neural model as the main deployable model unless it beats the protected policy under the same safety rules.

## What I Am Not Claiming

The boundaries matter:

- this is not a true 3D world model;
- this is not a large-scale foundation world model;
- SDD results are pixel-space, not metric predictions;
- external results are dataset-local unless a source has verified calibration;
- `t+50` and `t+100` are raw annotation-frame horizons, not seconds-level claims;
- self-audited or inferred scene labels are not human gold labels;
- latent generative execution has not been enabled;
- SMC has not been enabled.

I keep these limits explicit because the project is meant to be useful research, not a collection of inflated claims.

## How To Read This Repository

| Path | What it is for |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | The experiment ledger: results, failures, claim boundaries, and current research state. |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | A detailed Chinese retrospective of the routes I tried, what failed, and what worked. |
| `outputs/m3w_neural_v1/` | Neural candidate reports, model cards, and related evidence. |
| `outputs/stage42_long_research/` | Source/domain, full-waypoint, safety, replay, and paper-claim evidence. |
| `outputs/stage43_latent_state/` | Protected latent-state, tail adapter, bounded residual, and reviewer replay evidence. |
| [`research_state.json`](research_state.json) | Machine-readable project state. |

Large raw datasets, derived caches, checkpoints, videos, images, and local virtual environments are intentionally not committed. The repository keeps code, configs, lightweight metrics, and auditable reports.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Basic test command:

```bash
.venv-pytorch/bin/python -m pytest tests
```

Training and evaluation scripts are written to support checkpointing, heartbeat logs, resume, CPU/MPS-safe execution, and single-process dataloading.

## Near-Term Direction

The next research push is to turn the current protected policy into a stronger world-model candidate without losing its safety guarantees:

1. replace proxy-heavy scene and interaction evidence with stricter retrained ablations;
2. strengthen weak long-horizon slices, especially raw-frame `t+100`;
3. keep training neural dynamics heads, but only promote them when they beat the protected policy rather than merely imitate it;
4. preserve the claim boundary until calibration, scale, and stronger cross-domain evidence justify saying more.

If a route fails, I keep the failure in the repo. That is part of the project.
