# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my long-running research project on real-world multi-agent world modeling.

The short version: I am trying to build a model that can read a real scene, understand how multiple agents move together, and predict future world-state trajectories safely enough to be useful. The current system is already much stronger than the early baselines on my external raw-frame 2.5D benchmark, but I am keeping the claims deliberately narrow: this is not a true 3D model, not a foundation model, and not a metric/seconds-calibrated system yet.

This repository contains the code, evaluation harnesses, reports, and reproducibility artifacts behind that work.

## Current Status

The current best line of work is:

**M3W-Neural v1 composite-tail safe-switch bounded neural dynamics**

It is protected by the Stage37 / teacher-floor policy. In plain language, the neural model is allowed to help only when the safety policy says the switch is likely to improve the prediction without hurting easy cases. If the model is uncertain or risky, it falls back to a strong causal baseline.

That protection is not a cosmetic detail. It is one of the main findings of the project so far: ungated neural predictions can look promising on average but become unsafe or unstable on easy / close-proximity cases.

## What The Model Currently Does

M3W currently works as a protected 2.5D multi-agent world-state model:

- predicts future agent trajectories in top-down / trajectory-style datasets
- uses causal history windows, neighbor context, goal/prototype context, source/domain information, and baseline rollouts
- compares neural proposals against strong causal / teacher-floor predictions
- uses safe-switch and bounded correction policies rather than unbounded residuals
- evaluates all-agent waypoint predictions, endpoint behavior, hard/failure subsets, easy-case preservation, and proximity/smoothness proxies

The strongest evidence at the moment supports a **protected dataset-local / raw-frame 2.5D world-state dynamics candidate**.

## What It Does Not Claim

I am intentionally not claiming the following:

- not a true 3D world model
- not a large-scale foundation world model
- not a globally metric-calibrated prediction system
- not seconds-level long-horizon prediction
- not ungated neural deployment
- not Stage5C latent generative execution
- not SMC readiness

The current benchmarks are raw-frame / dataset-local. SDD is pixel-space. External datasets are also treated as dataset-local or weak/diagnostic unless homography, scale, FPS, and source terms are verified.

## Best Current Results

These are the headline results I currently trust enough to summarize publicly. They are still bounded by the raw-frame / dataset-local setup above.

| Model / policy | Role | Main evidence |
| --- | --- | --- |
| Stage26 cost-aware selector | SDD deployable baseline | improves SDD t+50 and hard/failure while preserving easy cases |
| Stage37 external t+50 safe selector | external safety floor | repairs external t+50 transfer with positive bootstrap CI |
| M3W-Neural v1 protected neural dynamics | current protected neural candidate | improves external all / t+50 / t+100 raw diagnostic / hard-failure under teacher-floor protection |
| Stage42 full-waypoint / group-consistency family | current world-state evidence | extends endpoint-style success toward all-agent full-waypoint prediction, still protected |

The strongest current public-facing claim is not "the model solves world modeling." It is:

> M3W is a protected raw-frame 2.5D multi-agent world-state modeling system that improves over strong causal and selector baselines on the current external trajectory benchmark, while preserving easy-case safety under a teacher-floor policy.

## Why The Safety Floor Matters

A large part of this project has been learning what not to deploy.

Several neural variants improved some aggregate numbers but failed on easy cases, t+100 slices, proximity, or domain transfer. That is why the current system uses a conservative protected policy:

1. compute strong causal / teacher-floor rollouts
2. let neural models propose endpoint or full-waypoint improvements
3. estimate gain, harm, uncertainty, and proximity risk
4. switch only when the validation-selected guard allows it
5. otherwise keep the safe baseline

This is not just a fallback hack. Right now it is part of the method.

## Research Direction

The long-term goal is bigger than the current selector/protected-policy system. I want M3W to become a genuinely useful multimodal multi-agent world model.

The next research questions are:

- Can full-waypoint sequence dynamics beat endpoint-to-linear bridge policies more broadly?
- Can scene, goal, neighbor, and interaction context become independent positive contributors rather than auxiliary diagnostics?
- Can I add legally verified external top-down datasets and reduce source concentration?
- Can any subset be calibrated to metric coordinates or seconds-level horizons without overclaiming?
- Can the model reduce dependence on the Stage37 / teacher floor without losing safety?

If those are not solved, the honest conclusion remains: strong protected 2.5D evidence, but not a true 3D or foundation-track world model.

## Repository Map

The repo is large because I keep the research history and negative results. The most useful entry points are:

| Path | Purpose |
| --- | --- |
| `src/` | data processing, models, evaluation, gates, report builders |
| `configs/` | training / evaluation configs |
| `tests/` | regression tests and evidence checks |
| `outputs/m3w_neural_v1/` | current M3W-Neural v1 summaries and model cards |
| `outputs/stage42_long_research/` | Stage42 evidence package, gates, ablations, claim guards |
| `README_RESULTS.md` | full internal experiment ledger |
| `research_state.json` | machine-readable project state |

Raw data, large caches, checkpoints, and third-party datasets are intentionally not committed.

## Reproducibility

The local training environment I use on Apple Silicon is:

```bash
.venv-pytorch/bin/python
```

Important runtime choices:

- arm64 Python / PyTorch environment
- `num_workers = 0`
- CPU-safe and MPS-safe paths where applicable
- checkpoint / heartbeat / resume support for long runs

Typical verification command:

```bash
.venv-pytorch/bin/python -m pytest tests
```

The current test suite is large because many reports and gates are checked as part of reproducibility.

## Reading The Results

If you only want the latest story, start with:

- `outputs/m3w_neural_v1/README_M3W_NEURAL_V1.md`
- `outputs/stage42_long_research/paper_claim_contract_stage42.md`
- `outputs/stage42_long_research/paper_contract_compliance_stage42.md`
- `README_RESULTS.md`

I keep both positive and negative results because they matter here. Some examples of negative or bounded findings:

- JEPA has not yet become an independent main contribution.
- Transformer-only / ungated neural variants are not currently deployable.
- Scene/goal and neighbor/interaction features are not yet stable independent main claims under the current protocol.
- Metric/time claims remain blocked until source terms, FPS/stride, homography, and scale are verified.

## Current Verdict

M3W is currently a strong protected 2.5D multi-agent world-state candidate, not the final world model I want.

The project has crossed an important line: it is no longer just a baseline selector demo. It has protected neural dynamics, full-waypoint / group-consistency evidence, safety-floor analysis, reproducibility checks, and a paper-ready claim boundary. But the main open problems are still real: broader external validation, metric/time calibration, stronger independent module contributions, and safer floor relaxation.

That is the work I am continuing here.
