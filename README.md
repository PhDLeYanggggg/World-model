# M3W

M3W is my research project on real-world multimodal, multi-agent world modeling.

The current question is deliberately concrete:

> Given a top-down scene and the recent motion of multiple agents, can the model predict what is likely to happen next, improve the hard cases, and avoid damaging the easy ones?

I care about that last part. In trajectory forecasting it is easy to make a model look better on selected difficult examples while quietly making ordinary motion worse. A lot of this repository is therefore about causal inputs, no-leakage audits, conservative switching, and negative results.

## Current State

M3W is currently a protected 2.5D multi-agent world-state model. It is not a true 3D world model and it is not a foundation model.

The strongest deployable system is still safety-first: learned components estimate failure risk, expected gain, switch harm, long-horizon drift, and interaction context, but deployment falls back to a causal baseline when those signals are not trustworthy enough.

The latest protected latent-state work is promising, but it is not a blank check for neural rollout. A unit-consistent safe-switch policy improved external dataset-local raw-frame results while keeping the easy-case guard intact. A source-level audit also found weak slices, so I describe the result as a protected domain-level candidate with source-level caveats, not as uniform cross-source success.

## Evidence So Far

The useful progress has come from guarded policies rather than unconstrained generation:

- On SDD, a cost-aware selector improved the pixel-space raw-frame benchmark while keeping easy cases under control.
- On external top-down pedestrian data, causal history windows and scene-agnostic goal prototypes repaired the earlier `t+50` transfer failure.
- Later source/domain-aware policies improved external raw-frame performance under conservative fallback.
- Protected latent-state experiments now show neural dynamics signal, but only the unit-consistent, easy-safe switch is deployable.

The short version: M3W has real protected world-state behavior. The protection is part of the model design, not decoration.

## What Did Not Work

Several directions failed clearly enough to change the project:

- Hard classification of the "best baseline" switched too aggressively and hurt easy trajectories.
- JEPA-style representation learning avoided collapse, but has not yet become a reliable downstream driver.
- SDD-to-external zero-shot transfer failed before coordinate, horizon, and goal-context repair.
- Latent alignment reduced distribution distance without automatically improving prediction.
- Ordinary residual correction was not safe enough to deploy.
- Ungated Transformer and Hybrid dynamics did not beat the protected floor.
- A source-level latent model looked strong in normalized space, but a unit-consistent audit exposed easy-case harm.

Those failures are useful. They keep the project honest about what is actually working.

## What I Am Not Claiming

The current boundary is strict:

- SDD results are pixel-space benchmark results.
- External results are dataset-local raw-frame results unless a source has verified timing and geometry.
- `t+50` and `t+100` are raw-frame horizons, not seconds-level claims.
- Inferred scene, goal, and visual-prior labels are not human gold labels.
- Stage5C latent generative execution is not enabled.
- SMC is not enabled.

M3W is moving toward stronger multimodal world modeling, but I would rather state the boundary plainly than make the project sound more finished than it is.

## Repository Map

The GitHub front page is intentionally short. The detailed evidence is kept in the result files:

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

The next step is not to make the README louder. It is to earn stronger evidence:

- repair weak source slices without test-threshold tuning;
- audit timing, geometry, and scale source by source;
- improve raw-frame `t+100` without pretending it is seconds-level;
- train neural dynamics that beat the protected selector under the same safety rules;
- keep ablations clean for scene, goal, interaction, and latent-state contributions.

If M3W becomes a stronger world model, it should be because the experiments survive these checks.
