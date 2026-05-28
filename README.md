# M3W

I am building M3W as a real-world multimodal, multi-agent world-modeling project.

The core question is the one I keep coming back to:

> From a scene and the recent motion of the agents in it, can we predict what happens next without peeking into the future, and without making the easy cases worse?

That sounds simple, but most of the work is in making the answer honest. This repository is where I convert messy trajectory data, build strong causal baselines, test transfer across datasets, audit leakage, keep negative results, and only promote a model when it survives the safety gates.

## Current Status

M3W is not a finished 3D world model and it is not a foundation model. Right now it is best described as a protected 2.5D multi-agent world-state system.

The strongest results are conservative selector policies. They start from causal motion baselines, estimate when another baseline or learned head is likely to help, and fall back when the switch looks risky. That approach has been more reliable than trying to replace the baseline with an unrestricted neural model.

The important boundaries are:

- SDD results are pixel-space results.
- External top-down trajectory results are dataset-local unless geometry has been verified separately.
- `t+50` and `t+100` are raw-frame horizons, not seconds-level claims.
- Inferred scene, goal, and visual-prior labels are not human gold labels.
- Stage5C latent generative rollout has not been executed.
- SMC is not enabled.

I want this project to grow into a stronger multimodal agent-scene world model, but I do not want the README to overstate what the current evidence supports.

## What Works

The most reliable part of the project is the fallback-protected selector family.

On SDD, the best current policy improves the pixel-space trajectory benchmark while preserving easy cases. On external top-down pedestrian data, the strongest policies use causal history, goal prototypes, source-aware validation, and conservative safety gates to avoid unsafe switching. Recent source/domain experiments also improved protected full-waypoint and long-horizon raw-frame evidence, but only inside the stated dataset-local boundary.

The practical lesson so far is clear: strong baselines are not an embarrassment here. They are the floor that lets the learned pieces be useful without becoming reckless.

## What Did Not Work

Several directions were worth trying but are not current main claims:

- Hard one-label baseline classification switched too aggressively and damaged easy cases.
- JEPA-style representation learning avoided collapse, but has not yet produced reliable downstream lift.
- Zero-shot SDD-to-external transfer failed before coordinate, horizon, and history repair.
- Latent distribution alignment reduced domain distance without consistently improving prediction.
- Ordinary residual correction was not safe enough to deploy.
- Ungated Transformer and Hybrid dynamics did not beat the protected floor.
- Scene/goal and neighbor/interaction features are useful diagnostics, but not yet independent main contributions.

Those failures are part of the project. They explain why the current system is guarded instead of flashy.

## How To Read This Repository

The root README is intentionally short. It is the public introduction.

For the detailed research record, use:

| Path | What it is for |
| --- | --- |
| `README_RESULTS.md` | long experiment ledger and stage-by-stage evidence |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of what worked, what failed, and why |
| `outputs/m3w_neural_v1/` | current model cards, reports, and evidence files |
| `outputs/stage42_long_research/` | long-run audits, ablations, gates, and source/domain reports |
| `research_state.json` | machine-readable current state |

Large local data, caches, checkpoints, videos, images, and third-party raw files are not committed.

## Running Locally

On Apple Silicon, I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training/evaluation code is written to avoid the old x86_64 Conda + Intel OpenMP path:

- `num_workers = 0`;
- CPU-safe and MPS-safe execution paths where applicable;
- checkpoints, heartbeat files, and resume support for long runs.

Typical verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Direction

The next useful progress is not bigger wording. It is better evidence:

- broader external top-down validation;
- safer long-horizon raw-frame transfer;
- stronger source and held-out-scene coverage;
- neural dynamics that beat the protected selector rather than copy it;
- metric or seconds-level reporting only after timing and geometry audits justify it.

That is the standard I am holding M3W to: promising, guarded, unfinished, and measured by what survives audit.
