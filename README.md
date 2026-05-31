# M3W: Real-World Multimodal Agent-Scene World Model

This repository is where I am building M3W, a research project about real-world multi-agent world modeling from top-down scenes, trajectories, and scene context.

The question I keep coming back to is simple:

> Can a model use only causal scene and motion history to predict what agents will do next, help on genuinely hard cases, and leave easy cases alone?

That last requirement matters. A model can look impressive if it only improves difficult slices while quietly making normal motion worse. Most of the work here is about avoiding that trap: no future leakage, no test-endpoint goal construction, strong causal baselines, conservative fallback, and explicit negative results.

## Where the Project Is Now

M3W is currently a protected 2.5D multi-agent world-state model. It is not a true 3D world model, and it is not a foundation model.

The best deployable versions are safety-first systems. Learned components estimate things like failure risk, expected gain, switch harm, long-horizon drift, interaction context, and latent world-state signals. They are only allowed to act when the evidence is strong enough; otherwise the system falls back to a causal baseline.

That fallback is not an afterthought. It is part of the model. The goal is not to make a neural component win a leaderboard slice once, but to build a world model that knows when not to intervene.

## What Has Worked

The strongest evidence so far is protected rather than unconstrained:

- On SDD, a cost-aware selector improved the pixel-space raw-frame benchmark while keeping easy cases under control.
- On external top-down pedestrian data, causal history windows and scene-agnostic goal prototypes fixed an earlier `t+50` transfer failure.
- Source/domain-aware policies improved external dataset-local raw-frame results under conservative fallback.
- Protected latent-state experiments show useful neural dynamics signal, but only after unit-consistent safety checks and easy-case guards.

The honest version is: M3W has real protected world-state behavior, with source-level caveats. I do not describe it as uniform cross-source success.

## What Failed

A lot of the project has been useful because it failed:

- A hard classifier for the "best baseline" switched too aggressively and damaged easy trajectories.
- JEPA-style representation learning did not collapse, but it has not yet become a reliable downstream driver.
- Direct SDD-to-external transfer failed before coordinate, horizon, and goal-context repair.
- Latent alignment reduced distribution distance without automatically improving prediction.
- Ordinary residual correction was not safe enough to deploy.
- Ungated Transformer and Hybrid dynamics did not beat the protected floor.
- A normalized-space source-level latent model looked strong until a unit-consistent audit exposed easy-case harm.

I keep these failures visible because they define the real boundary of the work.

## Claim Boundary

For now, the claims are deliberately narrow:

- SDD results are pixel-space benchmark results.
- External results are dataset-local raw-frame results unless timing and geometry are verified for that source.
- `t+50` and `t+100` are raw-frame horizons, not seconds-level claims.
- Inferred scene, goal, and visual-prior labels are not human gold labels.
- Latent generative execution is not enabled.
- SMC is not enabled.

M3W is moving toward stronger multimodal world modeling, but I would rather understate the result than make the project sound more finished than it is.

## Reading the Repository

This front page is intentionally short. The detailed evidence is kept in result files:

| Path | Purpose |
| --- | --- |
| `README_RESULTS.md` | Main running evidence ledger |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of routes tried, failures, and successes |
| `outputs/m3w_neural_v1/` | M3W neural model cards and reports |
| `outputs/stage42_long_research/` | Long-run audits, ablations, gates, and source/domain reports |
| `outputs/stage43_latent_state/` | Protected latent-state experiments and caveat audits |
| `research_state.json` | Machine-readable project state |

Large datasets, caches, checkpoints, videos, third-party raw data, and local virtual environments are intentionally not committed.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training setup is intentionally conservative:

- `num_workers = 0`;
- checkpoint and heartbeat support for long runs;
- resume support where training is long;
- CPU/MPS-safe execution;
- no x86_64 Conda + Intel OpenMP training path.

Basic verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Next

- repair weak source slices without test-threshold tuning;
- audit timing, geometry, and scale source by source;
- improve raw-frame `t+100` without pretending it is seconds-level;
- train neural dynamics that beat the protected selector under the same safety rules;
- keep ablations clean for scene, goal, interaction, and latent-state contributions.

If M3W becomes a stronger world model, it should be because the experiments survive these checks.
