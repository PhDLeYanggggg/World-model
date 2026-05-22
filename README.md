# World Model Lab

World Model Lab is an experimental 2.5D / pseudo-3D world-model research repo for
physical trajectory prediction, scene grounding, and causal evaluation.

The project started as a small Matter.js physical sandbox and has grown into a
staged research pipeline for pedestrian, crowd, and traffic world-state modeling:
dataset conversion, scene and goal annotation, multi-agent episode building,
strong causal baselines, deterministic residual models, failure mining, and
strict gates that prevent overclaiming.

This is not a true 3D foundation world model yet. The current system is best
described as a deterministic per-agent multi-agent bounded residual model over
the strongest causal baseline.

```text
prediction_i = baseline_i + alpha_i * bounded_residual_i
```

## Current Status

Current verified stage: Stage 13.

Stage 13 executed the first overnight deterministic repair loop:

- training/search trials executed: 24
- best model family: `residual_no_alpha`
- Stage 13 gates passed: 5 / 12
- expert audit score: 84 / 100
- best hard/failure improvement: 0.013127 on ETH/UCY EWAP t+50
- latent generative Stage 5C: blocked
- SMC: blocked

The honest verdict is:

```text
stage13_deterministic_repair_loop_executed_not_stage5c_ready
```

Stage 12 fixed a major data blocker by adding a verified pedestrian long-horizon
source (`eth_ucy_ewap`) with t+50 / t+100 coverage. Stage 13 then exposed a
stricter model-evaluation blocker: EWAP t+100 has no evaluable rows under the
Stage 13 per-agent causal mask. For that reason, this repo must not claim
pedestrian t+100 improvement yet.

See:

- [README_RESULTS.md](README_RESULTS.md)
- [Stage 13 final report](outputs/reports/report_stage13_final.md)
- [Stage 13 gates](outputs/reports/world_model_gate_stage13.md)
- [Research state](outputs/reports/research_state.md)

## What This Repo Builds

- A browser-based 2D physical sandbox for debugging interactions and JSONL-style
  episode export.
- Synthetic and real trajectory loaders for traffic, pedestrian, and
  drone-like/crowd datasets.
- Scene packs with walkable priors, candidate goals, annotation metadata, and
  train-only endpoint-derived goal regions.
- Multi-agent and per-agent episode builders with causal masks.
- Strong causal baselines, leakage audits, hard/failure subset mining, and
  long-horizon gates.
- Deterministic residual world models with scene, goal, and interaction
  ablations.
- An auto-orchestrator that reads gates, runs bounded repair loops, and writes
  reports without authorizing latent training or SMC prematurely.

## Repository Map

```text
src/data/          dataset loaders and unified episode builders
src/scene/         scene packs, SDFs, route graphs, walkable areas, goals
src/models/        deterministic residual, failure, goal, and encoder models
src/training/      stage-specific training and evaluation routines
src/evaluation/    gates, benchmarks, audits, failure mining, GoalBench
src/annotation/    human/silver annotation tools and export helpers
src/orchestrator/  auto-loop planning, gate reading, and reporting
configs/           stage and overnight-loop configuration files
outputs/reports/   generated model cards, gate reports, and summaries
```

## Quick Start

Install the Python research dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the current gate checks and inspect the latest state:

```bash
python run_stage13_gates.py
python run_stage13_failure_miner.py
python run_world_model_audit.py
```

Run the auto-orchestrator loop:

```bash
python run_auto_world_model_loop.py
```

Run the Stage 12 data pipeline if the required local data is present:

```bash
python run_stage12_pipeline.py
```

## Browser Sandbox

The frontend sandbox is a Vite + React + Matter.js app for interactive physical
world debugging and episode export.

```bash
npm install
npm run dev
```

Then open the local Vite URL printed by the dev server.

## Data Notes

The repo expects some datasets to exist locally for full reproduction. Several
large raw data folders, checkpoints, and generated artifacts are intentionally
ignored by Git. Public reports under `outputs/reports/` record what was loaded,
what passed gates, and what remains blocked.

Current Stage 12 data summary:

- loaded pedestrian/drone sources: `eth_ucy_ewap`, `aerialmpt`,
  `full_trajnet_original_quick`
- verified t+50 / t+100 pedestrian source: `eth_ucy_ewap`
- multi-agent episodes with at least two agents: 660
- hard/failure records: 649
- GoalBench v4 official records: 5574

AerialMPT remains pixel-space unless a homography or metric scale is provided.
ETH/UCY EWAP uses metric coordinates with `dt = 0.4s`; t+100 is about 40 seconds.

## Guardrails

The project intentionally fails loudly when the evidence is not strong enough.

- Do not enable latent generative Stage 5C until deterministic gates pass.
- Do not enable SMC until a stochastic proposal improves coverage.
- Do not claim pedestrian t+100 improvement until EWAP t+100 is evaluable under
  the per-agent causal mask.
- Do not treat rule-confirmed or AI visual-silver labels as human gold labels.
- Do not use test endpoints to construct candidate goals.

## Next Research Fixes

1. Fix or rebuild EWAP t+100 episode construction so per-agent causal past and
   t+100 target masks are evaluable.
2. Continue deterministic repair with stronger fallback-to-baseline discipline
   and hard/failure weighting.
3. Add more verified pedestrian/drone long-horizon data, especially SDD or
   OpenTraj, when legal local paths are available.
