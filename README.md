# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on real-world multi-agent world modeling from top-down pedestrian and agent-scene data.

The core problem is not just forecasting a trajectory. I am trying to build a model that understands enough about motion history, scene structure, goals, and interaction to make useful future-state predictions while staying honest about what it knows.

In practice, that means every learned component has to compete with strong causal baselines, survive no-leakage audits, and avoid breaking easy cases just to improve hard ones.

## The Research Question

Can a model use only past motion, scene context, and agent interactions to predict future multi-agent behavior better than a carefully protected causal baseline?

That question sounds narrow, but it is the part I care about. A world model that needs future endpoints, central velocities, test-set goals, or optimistic metric assumptions is not useful here. I would rather keep a smaller claim that survives audit than a bigger claim that depends on hidden leakage.

## Current State

M3W is currently a protected 2.5D / pseudo-3D multi-agent world-state model.

It is not a true 3D world model. It is not a large-scale foundation world model. SDD results are pixel-space benchmark results, and external pedestrian results are dataset-local raw-frame results unless timing and geometry have been verified for that source.

The strongest deployed path today is safety-first:

- a strong causal baseline is always available;
- learned policies estimate failure risk, expected gain, switch harm, long-horizon drift, interaction context, and latent world-state signals;
- the learned model is allowed to act only when validation-selected safety rules support the switch;
- otherwise the system falls back to the causal baseline.

That fallback is not an excuse. It is part of the model design. The goal is to improve difficult cases without quietly damaging normal motion.

## What Has Worked So Far

The clearest progress is in protected raw-frame settings.

On SDD, cost-aware selection improved the pixel-space benchmark while keeping easy-case degradation controlled. On external top-down pedestrian data, causal history windows and scene-agnostic goal prototypes repaired an earlier `t+50` transfer failure. Later source-aware and horizon-aware policies made the external transfer safer, and full-waypoint latent dynamics improved full-trajectory prediction under guards.

The current best results are useful, but they are still conditional. They are strongest when the safety floor is active, source and horizon caveats are respected, and the claim stays in raw-frame / dataset-local space.

## What Has Failed

I keep the failures visible because they define the real boundary of the project.

- A hard classifier for "best baseline" switched too aggressively and damaged easy cases.
- JEPA-style representation learning avoided collapse, but has not yet become a reliable standalone downstream driver.
- Direct SDD-to-external transfer failed before coordinate, horizon, and goal-context repair.
- Latent alignment sometimes reduced distribution distance without improving prediction.
- Ordinary residual correction was not safe enough to deploy.
- Ungated Transformer and Hybrid dynamics did not beat the protected floor.
- Raw-frame `t+100` remains the hardest open horizon.

Those failures are not side notes. They are the reason the current system is conservative.

## What I Do Not Claim

The wording in this repository is intentionally strict.

- I do not claim true 3D prediction.
- I do not claim foundation-model scale.
- I do not treat SDD pixel coordinates as metric coordinates.
- I do not describe raw-frame `t+50` or `t+100` as seconds-level horizons unless timing has been audited.
- I do not treat inferred scene or goal labels as human gold labels.
- I have not enabled latent generative execution.
- I have not enabled SMC.

M3W is a serious world-modeling track, but it is still a protected 2.5D multi-agent world-state candidate.

## How To Read The Repository

This README is the public entry point. The detailed evidence is kept in the result ledgers and stage reports.

| Path | Purpose |
| --- | --- |
| `README_RESULTS.md` | Main evidence ledger |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of routes tried, failures, and successes |
| `outputs/m3w_neural_v1/` | Neural model reports and model cards |
| `outputs/stage42_long_research/` | Long-run audits, ablations, gates, and source/domain reports |
| `outputs/stage43_latent_state/` | Protected latent-state experiments and caveat audits |
| `research_state.json` | Machine-readable project state |

Large datasets, derived caches, checkpoints, videos, third-party raw data, and local virtual environments are intentionally not committed.

## Local Setup

On Apple Silicon, I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training path is deliberately conservative:

- `num_workers = 0`;
- checkpoint and heartbeat support for long runs;
- resume support for longer experiments;
- CPU/MPS-safe execution;
- no x86_64 Conda + Intel OpenMP training path.

Basic verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Next Steps

The next useful work is not to make the README sound bigger. It is to make the evidence stronger:

- repair weak source and horizon slices without test-threshold tuning;
- audit timing, geometry, and scale source by source;
- improve raw-frame `t+100` without calling it seconds-level prediction;
- train neural dynamics that beat the protected selector under the same safety rules;
- keep ablations clean for scene, goal, interaction, latent-state, and fallback contributions.

If M3W becomes a stronger world model, it should be because the experiments survive those checks.
