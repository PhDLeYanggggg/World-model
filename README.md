# M3W

M3W is my long-running research project on real-world multimodal, multi-agent world modeling.

The short version: I am trying to build a model that can look at top-down agent motion, local scene context, nearby agents, and simple causal physics baselines, then decide what future behavior is plausible without cheating by looking into the future.

The longer version is more interesting. A lot of trajectory models improve the average number, but still fail in the places I care about: they switch away from a safe baseline on easy cases, break when the coordinate system changes, or learn a latent space that looks nice but does not actually help downstream prediction. M3W is my attempt to make that failure visible and then build around it.

## Where The Project Stands

The strongest deployable piece right now is not a huge end-to-end neural model. It is a protected prediction policy:

- start from strong causal motion baselines;
- use past-only history, neighbor, density, and goal-prototype features;
- estimate when switching is likely to help;
- fall back when the model is uncertain or the sample looks easy;
- keep leakage checks and hard-slice reports attached to every claim.

That conservative setup has been the most reliable path so far. It has beaten simpler baselines on SDD and on external top-down pedestrian data under dataset-local, raw-frame evaluation. The neural world-model track is still active, but I only promote a neural model when it beats the protected policy under the same safety rules.

## What This Repo Is

This is a research workspace, not a polished product demo.

I use it to keep the whole trail: the working ideas, the failed routes, the ablations, the leakage audits, the reports, and the current best deployable candidate. If something fails in a useful way, I leave it in. That is how I keep the project honest.

The most useful files to start with are:

| Path | Why it matters |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | Main results ledger: current metrics, negative results, and claim boundaries. |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | Detailed Chinese summary of what I tried, what failed, and what worked. |
| `outputs/m3w_neural_v1/` | Neural-model reports and model-card style summaries. |
| `outputs/stage42_long_research/` | Cross-domain, safety, replay, and paper-claim evidence. |
| `outputs/stage43_latent_state/` | Latent-state experiments, feature-family ablations, and reviewer-style evidence. |
| [`research_state.json`](research_state.json) | Machine-readable snapshot of the current research state. |

Large raw datasets, derived caches, checkpoints, videos, images, and local virtual environments are deliberately left out of Git.

## What I Am Not Claiming

I am careful about this because inflated world-model claims are too easy to make:

- M3W is not a true 3D world model yet.
- M3W is not a foundation world model.
- SDD results are pixel-space results, not metric predictions.
- External results are dataset-local unless calibration is verified.
- `t+50` and `t+100` are raw annotation-frame horizons, not seconds.
- Self-audited or inferred scene labels are not human gold labels.
- Stage5C latent generative execution has not been enabled.
- SMC has not been enabled.

So the current claim is narrower: this repo contains a serious, auditable path toward a safer multimodal agent-scene world model, with a protected selector as the strongest deployable base and neural dynamics still under active testing.

## Running It Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Basic test command:

```bash
.venv-pytorch/bin/python -m pytest tests
```

The training and evaluation scripts are written around checkpointing, heartbeat logs, resume, CPU/MPS-safe execution, and single-process dataloading.

## What I Am Working On Next

The next step is to make the neural world-model side earn its place:

1. keep the protected policy as the safety floor;
2. retrain scene, goal, and interaction ablations instead of relying on proxy-only evidence;
3. improve weak long-horizon slices, especially raw-frame `t+100`;
4. only promote neural dynamics when they beat the protected policy rather than simply imitate it.

If the neural route fails, the repo should say that plainly. If it works, it should be clear exactly which part worked and under what evaluation boundary.
