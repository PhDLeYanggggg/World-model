# M3W

**Real-World Multimodal Agent-Scene World Model**

M3W is my research project on multi-agent world modeling from real top-down pedestrian and agent-scene data. The question I care about is simple to state and hard to do well:

> Can a model use only past motion, scene context, and local interactions to predict future multi-agent behavior better than a strong causal baseline, without cheating and without damaging easy cases?

Most forecasting systems look better if they are allowed to lean on future endpoints, test-set statistics, optimistic geometry, or weak baselines. I am deliberately building M3W under stricter rules. The model only gets past information at inference time. It has to beat protected causal policies on useful slices, preserve easy examples, and survive no-leakage audits.

## The Idea

M3W treats world modeling as an agent-scene problem, not just a curve-fitting problem. The system combines:

- causal motion history;
- local density and neighbor interaction features;
- scene and goal context when it can be built without test leakage;
- strong physical baselines;
- learned failure, gain, harm, and switchability models;
- conservative fallback policies for cases where learning is not reliable enough.

The current philosophy is safety-first: a learned component is only useful if it can say when it should not be trusted. If the model cannot beat the fallback under validation-selected rules, it does not get deployed.

## Where The Project Stands

M3W is currently a protected 2.5D multi-agent world-state model. It is not a true 3D world model, and it is not a foundation model.

The strongest results so far are protected policies that improve hard or longer-horizon raw-frame cases while keeping easy-case degradation small. The latest work also includes neural and latent-state experiments, but those components remain behind a safety floor unless they prove they can improve the deployed policy.

The important point is that I do not treat every positive experiment as a deployable model. Some results are only diagnostics. Some are useful for understanding the failure modes. The deployable path stays conservative until the evidence is strong enough.

## What Has Worked

The lines of work that have held up best are:

- cost-aware baseline selection instead of hard best-class classification;
- confidence-gated fallback rather than unconditional switching;
- causal history windows for external top-down transfer;
- scene-agnostic goal prototypes built only from training data;
- source-aware and horizon-aware policies;
- protected latent-state and bounded-residual policies that keep a causal safety floor.

The detailed numbers, gates, replay checks, and bootstrap reports live in `README_RESULTS.md` and `outputs/`. I keep the GitHub front page lighter so it reads like a project introduction rather than an experiment ledger.

## What Has Not Worked Yet

Several routes have failed or stayed diagnostic:

- hard classification of the oracle-best baseline switched too aggressively;
- JEPA-style pretraining has avoided collapse but has not consistently improved downstream heads;
- raw SDD-to-external zero-shot transfer failed before coordinate, horizon, and goal-context repair;
- latent distribution alignment reduced distance without always improving prediction;
- ordinary residual correction was not safe enough without a fallback;
- unprotected Transformer or Hybrid dynamics did not replace the protected policy;
- raw-frame `t+100` is still an open problem.

Those failures are part of the project. I keep them because they show what the model is not yet able to do.

## Claim Boundaries

I am intentionally conservative about what I claim.

- SDD results are pixel-space unless a source-specific calibration says otherwise.
- External pedestrian results are dataset-local/raw-frame unless timing, homography, and scale are verified.
- Raw-frame `t+50` and `t+100` are not seconds-level claims.
- Self-audited or inferred scene labels are not human gold labels.
- Stage5C latent generative execution has not been run.
- SMC is not enabled.

The current project is a real step toward multimodal multi-agent world modeling, but it is still a protected dataset-local 2.5D world-state candidate.

## How To Read This Repository

The public README is the short version. The evidence is stored in the research ledgers and reports.

| Path | What it contains |
| --- | --- |
| `README_RESULTS.md` | Main experiment ledger and high-level results |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese summary of routes tried, successes, failures, and causes |
| `outputs/m3w_neural_v1/` | Neural model reports and model cards |
| `outputs/stage42_long_research/` | Long-run audits, ablations, replay reports, and source/domain analysis |
| `outputs/stage43_latent_state/` | Protected latent-state and bounded-residual evidence |
| `research_state.json` | Machine-readable project state |

Large datasets, generated caches, checkpoints, videos, third-party raw data, and local virtual environments are not committed.

## Local Notes

On Apple Silicon I run training in the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training path is designed to be recoverable:

- single-process data loading by default;
- checkpoint and heartbeat support;
- resume support for long experiments;
- CPU/MPS-safe execution;
- no x86_64 Conda + Intel OpenMP training path.

Basic test command:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Next

The next real milestone is to make the neural world dynamics carry more of the result, instead of relying so heavily on protected selection and fallback. That means improving weak source and horizon slices, auditing timing and geometry source by source, and proving through ablations that scene, goal, interaction, and latent-state signals add value under the same no-leakage rules.

If M3W becomes a stronger world model, it should be because the experiments earn the claim.
