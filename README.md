# M3W

M3W is a research project on real-world multimodal, multi-agent world modeling from top-down trajectory data.

The project started from a practical question:

> If I know what a scene looks like and how all nearby agents have been moving, can I predict what happens next without peeking into the future, and without making the easy cases worse?

That sounds simple, but in practice most of the work is not just the model. It is dataset conversion, leakage checks, causal baselines, external transfer, failure analysis, and keeping the claims honest when an experiment only works on one slice.

## What I Am Building

M3W is meant to become an agent-scene world model: a system that understands motion history, local interaction, scene context, possible goals, and safety risk well enough to forecast multi-agent futures.

The current version is not that full vision yet. Today it is best described as a protected 2.5D multi-agent world-state model. The strongest deployable pieces are conservative selector policies that sit on top of causal motion baselines. They only switch away from the safe baseline when the expected gain is high and the risk of harming easy cases is low.

That safety floor matters. In this project, a model that improves hard cases but breaks easy cases is not deployable.

## Current Claim Boundary

I keep the public claim narrow on purpose:

- this is not a true 3D world model;
- this is not a foundation model;
- SDD results are pixel-space benchmark results;
- external top-down results are dataset-local unless source-specific timing and geometry are verified;
- `t+50` and `t+100` mean raw-frame horizons, not seconds-level claims;
- inferred scene or goal labels are not human gold annotations;
- latent generative rollout and SMC are not enabled.

Those boundaries are part of the work, not fine print. I would rather have a smaller result that survives audit than a bigger story that depends on hidden leakage or unit confusion.

## What Works So Far

The most reliable progress has come from cost-aware selection with fallback:

- strong causal baselines are treated as part of the system;
- learned components estimate failure, gain, harm, and switchability;
- hard and long-horizon slices can improve when the policy is allowed to switch carefully;
- easy-case preservation is enforced instead of assumed;
- external transfer only counts when it survives held-out evaluation and no-leakage checks.

Recent experiments show useful protected gains on SDD and on external top-down pedestrian data in dataset-local/raw-frame form. The key word is protected: the learned parts are useful when they are guarded by a conservative policy floor.

## What Has Failed

Several routes were tried and are not current headline claims:

- hard one-label baseline classification switched too aggressively;
- JEPA-style representation learning avoided collapse but has not produced reliable downstream lift;
- zero-shot SDD-to-external transfer failed before coordinate and horizon repair;
- latent alignment reduced distribution distance without necessarily improving prediction;
- ordinary residual correction was not safe enough to deploy;
- ungated Transformer and Hybrid dynamics did not beat the protected floor;
- scene/goal and neighbor/interaction context are useful diagnostics, but not yet independent main contributions.

These negative results are useful. They are why the current model is guarded instead of presented as an unrestricted neural world model.

## Repository Guide

The root README is intentionally short. The detailed research record lives in separate files:

| Path | What it is for |
| --- | --- |
| `README_RESULTS.md` | experiment ledger and evidence notes |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of routes tried, failures, successes, and current quality |
| `outputs/m3w_neural_v1/` | model cards, data cards, and current M3W reports |
| `outputs/stage42_long_research/` | long-run audits, ablations, gates, and source/domain reports |
| `research_state.json` | machine-readable current state |

Large datasets, caches, checkpoints, video files, raw third-party data, and local virtual environments are intentionally not committed.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Training and evaluation paths are kept CPU/MPS-safe:

- `num_workers = 0`;
- checkpoint and heartbeat support for long runs;
- resume support where training is long;
- no x86_64 Conda + Intel OpenMP training path.

The basic verification command is:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Next Direction

The next useful progress is not a bigger claim. It is stronger evidence:

- broader external top-down validation;
- safer raw-frame long-horizon transfer;
- source-specific timing and geometry audits;
- neural dynamics that beat the protected selector instead of copying it;
- clearer ablations showing when scene, goal, interaction, and latent representations matter.

That is the standard I am using for M3W: promising, unfinished, and measured by what still works after the easy ways to fool myself are removed.
