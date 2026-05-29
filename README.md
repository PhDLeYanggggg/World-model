# M3W

I am building M3W as a research project on real-world multimodal, multi-agent world modeling.

The version in this repository is focused on a concrete question:

> From a top-down scene and the recent motion of nearby agents, can a model predict what is likely to happen next, improve the hard cases, and still leave the easy cases alone?

That last part matters. A model that looks better on difficult cases but quietly damages simple trajectories is not useful to me. A lot of this repo is therefore about safety floors, leakage checks, conservative switching, and negative results, not just new model code.

## Where The Project Stands

M3W is currently a protected 2.5D multi-agent world-state model. It is not a true 3D world model, and it is not a foundation model.

The strongest deployable pieces are conservative policies built on top of causal motion baselines. The learned parts estimate things like failure risk, possible gain, switch harm, long-horizon drift, and interaction/trajectory context. When those learned signals are not strong enough, the system falls back to the safer baseline.

That makes the current system less flashy than an unrestricted neural rollout, but it is much harder to fool myself with it.

## What Has Worked So Far

The clearest progress has come from protected world-state policies:

- On SDD, a cost-aware selector improved pixel-space raw-frame forecasting while keeping easy cases under control.
- On external top-down pedestrian data, a causal-history and goal-prototype policy repaired the earlier `t+50` transfer failure.
- Source/domain-aware policies later improved the external raw-frame results, but only under conservative fallback.
- Recent latent-state experiments show useful neural dynamics signal, but the safe floor still decides what is deployable.

So the honest short version is: M3W has promising protected multi-agent world-state behavior, but the protection is part of the model. It is not a temporary wrapper I can ignore.

## What Has Failed

The failures are important because they changed the direction of the project:

- A hard classifier for "best baseline" switched too often and hurt easy cases.
- JEPA-style representation learning avoided collapse, but has not yet become a reliable downstream contributor.
- Zero-shot SDD-to-external transfer failed before coordinate, horizon, and goal-context repair.
- Latent distribution alignment reduced domain distance without guaranteeing predictive value.
- Ordinary residual correction was not safe enough to deploy.
- Ungated Transformer and Hybrid dynamics did not beat the protected floor.
- A later source-level latent model looked strong in normalized space, but a unit-consistent audit showed easy-case harm, so it stayed non-deployable.

That is the current lesson: neural dynamics are worth pursuing, but only when they survive the same no-leakage and easy-preservation checks as the selectors.

## What I Am Not Claiming

These boundaries are intentional:

- SDD results are pixel-space benchmark results.
- External results are dataset-local unless timing and geometry are source-verified.
- `t+50` and `t+100` are raw-frame horizons, not seconds-level claims.
- Inferred scene, goal, and visual-prior labels are not human gold labels.
- Stage5C latent generative execution is not enabled.
- SMC is not enabled.

M3W is still a 2.5D / pseudo-3D trajectory world-state project moving toward stronger multimodal world modeling. I would rather keep the README modest than make the project sound more finished than it is.

## How To Read The Repository

The root README is the public introduction. The detailed evidence lives in the project ledgers and reports:

| Path | What it is for |
| --- | --- |
| `README_RESULTS.md` | running evidence ledger and important stage notes |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of what I tried, what failed, and what worked |
| `outputs/m3w_neural_v1/` | model cards, data cards, and M3W neural reports |
| `outputs/stage42_long_research/` | long-run audits, ablations, gates, and source/domain reports |
| `outputs/stage43_latent_state/` | protected latent-state experiments |
| `research_state.json` | machine-readable project state |

Large datasets, caches, checkpoints, videos, third-party raw data, and local virtual environments are intentionally not committed.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training setup is deliberately boring:

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

The next useful step is not a bigger claim. It is better evidence.

I am working on:

- broader external top-down validation;
- source-specific timing and geometry audits;
- safer raw-frame long-horizon transfer;
- neural dynamics that beat the protected selector under the same safety rules;
- cleaner ablations for scene, goal, interaction, and latent-state contributions.

If M3W becomes a stronger world model, it should be because it survived those checks, not because the README got ahead of the experiments.
