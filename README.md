# M3W

**Real-World Multimodal Agent-Scene World Model**

M3W is my long-running research project on multi-agent world modeling from top-down pedestrian and agent-scene data.

The project started from a simple question: can a model use only past motion, scene context, and local interactions to predict future multi-agent behavior better than strong causal baselines, without cheating through future endpoints or test-set information?

I am deliberately conservative about the claims here. The current system is not a true 3D world model, and it is not a foundation model. It is best described as a protected, dataset-local 2.5D world-state model. The point of the repository is to push that system forward while keeping the evidence honest.

## What This Repo Is For

M3W studies how far a real-world multi-agent model can go under strict evaluation rules:

- inference uses past information only;
- future endpoints are labels for training and evaluation, not inputs;
- test endpoints are never used to build goals;
- central velocity is not used as an official input;
- SDD remains pixel-space unless geometry is verified;
- external top-down data remains dataset-local unless timing, homography, and scale are verified.

That framing matters. A model that looks strong with leaked goals, optimistic scale assumptions, or weak baselines is not useful to me. I want improvements that survive strong fallback policies and no-leakage audits.

## Current State

The strongest deployable line so far is a protected policy stack: causal history features, interaction cues, train-only goal prototypes when available, horizon-specific switching, and conservative fallback to strong physical baselines.

This approach has worked better than hard "pick the oracle baseline" classification. The reliable policies estimate gain, harm, and failure risk, then switch only when the validation-selected safety rule says the learned component is worth trusting.

The neural and latent-state parts are active research. Transformer, JEPA, hybrid, residual, and latent-state heads are only treated as deployable when they improve the protected policy without breaking easy cases. When they fail, they stay in the reports as negative evidence.

Detailed metrics and stage-by-stage evidence live in [`README_RESULTS.md`](README_RESULTS.md). I keep the top-level README focused on what the project is and how to read it.

## What Has Worked

- Cost-aware baseline selection has been more reliable than hard best-baseline classification.
- Conservative fallback is essential; learned switching without a safety floor damages easy cases.
- External transfer did not work as zero-shot SDD transfer, but improved after causal history windows, scene-agnostic goal prototypes, and horizon-specific policies were added.
- Raw-frame `t+50` external transfer became useful only after the model learned when not to switch.
- Protected latent-state and bounded policies are promising, but they are still judged against the same safety floor.

## What Has Not Worked Yet

- JEPA-style representation learning has avoided collapse, but downstream lift is still inconsistent.
- Latent distribution alignment can make domains look closer without improving prediction.
- Unbounded residual correction is unsafe.
- Unprotected neural dynamics has not replaced the protected policy.
- Raw-frame `t+100` is still mostly diagnostic.
- ETH/TrajNet-style external coverage still needs stronger held-out validation before I would call the model broadly cross-domain.

These failures are not hidden. They are part of the research record.

## Claim Boundaries

When reading the results, I use these boundaries:

- SDD is pixel-space unless source-specific calibration proves otherwise.
- External data is dataset-local/raw-frame unless timing and scale are verified.
- Raw-frame `t+50` and `t+100` are not seconds-level horizons.
- Self-audited or inferred scene labels are not human gold labels.
- Latent-generative Stage5C has not been executed.
- SMC is not enabled.

The honest description today is: protected, dataset-local 2.5D multi-agent world-state modeling, with ongoing work toward stronger multimodal neural dynamics.

## Repository Map

| Path | Purpose |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | Main experiment ledger and high-level results |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | Chinese route-by-route summary of attempts, failures, and successes |
| `outputs/m3w_neural_v1/` | Neural model reports and model cards |
| `outputs/stage42_long_research/` | Long-run audits, ablations, replay reports, and source/domain analysis |
| `outputs/stage43_latent_state/` | Protected latent-state and bounded-residual evidence |
| [`research_state.json`](research_state.json) | Machine-readable project state |

Large datasets, generated caches, checkpoints, videos, third-party raw data, and local virtual environments are intentionally not committed.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training path is designed for long local runs:

- single-process data loading by default;
- checkpoint and heartbeat support;
- resume support;
- CPU/MPS-safe execution;
- no x86_64 Conda + Intel OpenMP training path.

Basic test command:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Next Milestone

The next milestone is to make neural world dynamics carry more of the result instead of relying so heavily on protected selection and fallback. That means improving weak source and horizon slices, auditing timing and geometry source by source, and proving through ablations that scene, goal, interaction, and latent-state signals add value under the same no-leakage rules.

I want M3W to earn stronger claims through evidence, not wording.
