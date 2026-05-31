# M3W: Real-World Multimodal Agent-Scene World Model

This repository is where I am building M3W, a research project on real-world multi-agent world modeling from top-down pedestrian and agent-scene data.

The question I keep coming back to is simple:

> Can a model look at only past motion, scene context, and agent interactions, then predict what happens next without cheating on the future or damaging easy cases?

The important part is the "without". A result only counts here if it survives strong causal baselines, no-leakage checks, conservative fallback, and slice-level audits. A model that improves a difficult slice while quietly breaking normal motion is not a useful world model.

## Where It Stands

M3W is currently a protected 2.5D / pseudo-3D multi-agent world-state model. It is not a true 3D world model, and it is not a large-scale foundation model.

The strongest version today is safety-first. Learned heads estimate failure risk, expected gain, switch harm, long-horizon drift, interaction context, and latent world-state signals. They are allowed to act only when validation-selected safety rules say the switch is worth it. Otherwise the system falls back to a strong causal baseline.

That fallback is part of the design, not a backup excuse. The practical goal is to improve hard motion cases while knowing when to leave easy cases alone.

## What Has Worked

The strongest evidence so far is in protected, raw-frame settings:

- On SDD, cost-aware selection improved the pixel-space benchmark while keeping easy-case degradation under control.
- On external top-down pedestrian data, causal history windows and scene-agnostic goal prototypes repaired the earlier `t+50` transfer failure.
- Source-aware and horizon-aware policies produced deployable external raw-frame gains under conservative fallback.
- Full-waypoint latent dynamics improved external full-trajectory prediction under safety guards, with strong `t+50` gains.

The honest summary: M3W has useful protected world-state behavior, but it still has source-level and horizon-level caveats. I do not claim uniform cross-source success.

## What Has Not Worked Yet

The negative results matter just as much:

- A hard "best baseline" classifier switched too often and damaged easy cases.
- JEPA-style representation learning avoided collapse, but has not yet become a reliable standalone downstream driver.
- Direct SDD-to-external transfer failed before coordinate, horizon, and goal-context repair.
- Latent alignment sometimes reduced distribution distance without improving prediction.
- Ordinary residual correction was not safe enough to deploy.
- Ungated Transformer and Hybrid dynamics did not beat the protected floor.
- Raw-frame `t+100` remains the hardest open horizon and is not yet a reliable positive result.

These failures are kept visible because they mark the real boundary of the current system.

## Claim Boundary

I am deliberately strict about the language in this repository:

- SDD results are pixel-space benchmark results.
- External results are dataset-local raw-frame results unless timing and geometry are verified for that source.
- `t+50` and `t+100` are raw-frame horizons, not seconds-level claims.
- Inferred scene, goal, and visual-prior labels are not human gold labels.
- Latent generative execution is not enabled.
- SMC is not enabled.

In short: this is a serious world-modeling track, but it is not yet a true 3D or foundation world model.

## How To Read This Repository

This front page is the plain-language entry point. The detailed evidence lives in the result ledgers:

| Path | What it contains |
| --- | --- |
| `README_RESULTS.md` | Main running evidence ledger |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of routes tried, failures, and successes |
| `outputs/m3w_neural_v1/` | Neural model reports and model cards |
| `outputs/stage42_long_research/` | Long-run audits, ablations, gates, and source/domain reports |
| `outputs/stage43_latent_state/` | Protected latent-state experiments and caveat audits |
| `research_state.json` | Machine-readable project state |

Large datasets, derived caches, checkpoints, videos, third-party raw data, and local virtual environments are intentionally not committed.

## Local Setup

On Apple Silicon, training should use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training path is intentionally conservative:

- `num_workers = 0`;
- checkpoint and heartbeat support for long runs;
- resume support for longer experiments;
- CPU/MPS-safe execution;
- no x86_64 Conda + Intel OpenMP training path.

Basic verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Next Research Steps

- repair weak source slices without test-threshold tuning;
- audit timing, geometry, and scale source by source;
- improve raw-frame `t+100` without calling it seconds-level prediction;
- train neural dynamics that beat the protected selector under the same safety rules;
- keep ablations clean for scene, goal, interaction, and latent-state contributions.

If M3W becomes a stronger world model, it should be because the experiments survive those checks, not because the README makes it sound finished.
