# M3W: Real-World Multimodal Agent-Scene World Model

I am building M3W as a research project on real-world multi-agent world modeling from top-down pedestrian and agent-scene data.

The motivation is simple: a useful world model should do more than extrapolate a line. It should understand motion history, local scene structure, goal intent, interaction pressure, and when a learned prediction is not trustworthy enough to beat a strong causal fallback.

This repository is the working record of that effort: code, experiments, audits, negative results, and the current best deployable policies.

## What M3W Is Trying To Do

The central question is:

Can a model use only past motion, scene context, and agent interactions to predict future multi-agent behavior better than a carefully protected causal baseline?

I care about the protected part. If a model needs future endpoints, central velocities, test-set goals, hidden normalization from test data, or optimistic geometry assumptions, then it is not solving the problem I want to solve.

The current system is built around a safety-first idea:

- keep a strong causal baseline available;
- learn when that baseline is likely to fail;
- estimate whether switching to a learned policy is worth the risk;
- only switch when validation-selected rules say the gain is likely and the harm is controlled;
- otherwise fall back.

That makes the project more conservative than a pure neural forecasting benchmark, but it also makes the claims easier to audit.

## Current Status

M3W is currently a protected 2.5D / pseudo-3D multi-agent world-state model.

It is not a true 3D world model. It is not a large-scale foundation world model. The SDD results are pixel-space results. External pedestrian results are dataset-local raw-frame results unless timing, geometry, and scale are verified for that source.

The strongest evidence so far is not that an unconstrained neural model wins everywhere. It is that protected policies can improve hard and long-horizon slices while keeping easy-case damage small.

The current best path is still conservative:

- Stage37 repaired external `t+50` transfer with causal history windows and scene-agnostic goal prototypes.
- Later work added source and horizon guards, latent-state experiments, and safety-floor audits.
- Neural dynamics and JEPA-style representations are being tested, but they are not allowed to replace the protected policy unless they beat it under the same safety rules.

## Results I Trust Most

The results I trust most are the ones that pass no-leakage checks and preserve easy cases.

The project has shown useful progress on:

- cost-aware baseline selection instead of hard "best baseline" classification;
- confidence-gated fallback policies;
- external `t+50` repair using past-only history windows;
- train-only goal prototypes for external top-down data;
- source-aware and horizon-aware transfer policies;
- audits showing when the safety floor is necessary.

These results are encouraging, but they are still raw-frame / dataset-local world-state results. I do not present them as metric 3D prediction or foundation-model behavior.

## What Has Not Worked Yet

I keep the failures in the repo because they are part of the research.

- A hard classifier for the best baseline switched too aggressively and damaged easy samples.
- JEPA-style representation learning avoided collapse, but has not consistently produced downstream lift.
- Direct SDD-to-external transfer failed before coordinate, horizon, and goal-context repair.
- Latent alignment sometimes reduced distribution distance without improving prediction.
- Ordinary residual correction was not safe enough to deploy.
- Ungated Transformer and Hybrid dynamics did not beat the protected safety floor.
- Raw-frame `t+100` remains a major open problem.

Those failures are why the current deployable path remains fallback-protected.

## Claims I Do Not Make

I am deliberately strict about the wording around this project.

- I do not claim true 3D prediction.
- I do not claim foundation-model scale.
- I do not treat SDD pixel coordinates as metric coordinates.
- I do not describe raw-frame `t+50` or `t+100` as seconds-level horizons unless timing has been audited.
- I do not treat inferred scene or goal labels as human gold labels.
- I have not enabled latent generative execution.
- I have not enabled SMC.

The current work is a serious step toward a real-world multimodal multi-agent world model, but it is still a protected 2.5D world-state candidate.

## Repository Map

The root README is intentionally short. The detailed evidence lives in the ledgers and reports.

| Path | What it is for |
| --- | --- |
| `README_RESULTS.md` | Main experiment and evidence ledger |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of routes tried, failures, and successes |
| `outputs/m3w_neural_v1/` | Neural model reports and model cards |
| `outputs/stage42_long_research/` | Long-run audits, ablations, gates, and source/domain reports |
| `outputs/stage43_latent_state/` | Protected latent-state experiments and caveat audits |
| `research_state.json` | Machine-readable project state |

Large datasets, derived caches, checkpoints, videos, third-party raw data, and local virtual environments are not committed.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training setup is intentionally simple and recoverable:

- `num_workers = 0`;
- checkpoint and heartbeat support for long runs;
- resume support for interrupted experiments;
- CPU/MPS-safe execution;
- no x86_64 Conda + Intel OpenMP training path.

Basic verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Where I Am Taking It Next

The next target is to make the neural world dynamics genuinely useful instead of decorative.

That means:

- improving weak source and horizon slices without tuning on test;
- auditing timing, geometry, and scale source by source;
- improving raw-frame `t+100` without calling it seconds-level prediction;
- training neural dynamics that beat the protected selector under the same safety rules;
- keeping ablations clean for scene, goal, interaction, latent-state, and fallback contributions.

If M3W becomes a stronger world model, it should be because the experiments earn the claim.
