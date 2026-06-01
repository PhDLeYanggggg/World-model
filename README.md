# M3W

**Real-World Multimodal Agent-Scene World Model**

M3W is my long-running research project on real-world multi-agent world modeling. I am interested in a practical question: if a model only sees past motion, scene context, and nearby agents, can it make better future predictions than strong causal baselines without relying on leaked future information?

The project is intentionally empirical. I build a model, compare it against conservative baselines, break down where it fails, and then tighten the next experiment around the failure mode. The repository is therefore both a model codebase and a research record.

## What I Am Building

M3W studies top-down multi-agent motion in real scenes. The current system combines:

- causal history features from agent trajectories;
- local interaction and density cues;
- train-only goal and route prototypes when they are available;
- strong physical baselines such as constant velocity and damped velocity;
- safety policies that only let learned components override a baseline when validation evidence says the switch is worth it;
- neural and latent-state experiments that are evaluated under the same safety rules.

The long-term goal is a stronger multimodal agent-scene world model. The current system is not there yet. Today it is best described as a protected, dataset-local 2.5D multi-agent world-state model.

## Current Best Evidence

The most reliable deployable direction so far is not a large end-to-end neural model. It is a protected policy stack that estimates failure risk, possible gain, and possible harm before deciding whether to switch away from a causal baseline.

That line of work has produced the strongest external transfer result so far:

- positive external all-test improvement;
- positive raw-frame `t+50` improvement;
- improved hard/failure cases;
- very small easy-case degradation;
- strict fallback to the strongest baseline when the learned policy is not confident.

Newer latent-state and neural dynamics experiments are promising in some slices, but I do not treat them as replacements unless they beat the protected policy while preserving easy cases. Negative results stay in the repo because they are useful: they show which ideas look good in isolation but do not survive deployment rules.

Detailed numbers, confidence intervals, and stage-by-stage reports are in [`README_RESULTS.md`](README_RESULTS.md).

## What Has Worked

- Cost-aware selection works better than hard oracle-label classification.
- Conservative fallback is necessary; learned policies without a safety floor hurt easy cases.
- External transfer became useful only after adding causal history windows, scene-agnostic goal prototypes, and horizon-specific safety rules.
- Raw-frame `t+50` is where the strongest deployable transfer evidence currently appears.
- Protected latent-state policies are a useful research direction, but they still need to pass the same safety and ablation checks.

## What Has Not Worked Yet

- JEPA-style representation learning has avoided collapse, but downstream lift is not yet stable enough to be a main claim.
- Latent distribution alignment can reduce domain distance without improving prediction.
- Unbounded residual correction is unsafe.
- Neural dynamics has not yet replaced the protected policy as the best deployable model.
- Raw-frame `t+100` remains mostly diagnostic.
- Broader ETH/TrajNet-style held-out validation still needs stronger coverage before I would call this broadly cross-domain.

## How To Read The Claims

I keep the claim boundaries explicit because they matter for this project.

SDD results are pixel-space unless source-specific calibration proves otherwise. External top-down results are dataset-local/raw-frame unless timing, homography, and scale are verified. Raw-frame `t+50` and `t+100` should not be read as seconds-level horizons. Self-audited or inferred scene labels are not human gold labels.

Latent-generative Stage5C has not been executed, and SMC is not enabled.

## Repository Map

| Path | Purpose |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | Main results ledger and experiment summary |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | Chinese summary of attempts, failures, and successes |
| `outputs/m3w_neural_v1/` | Neural model reports and model cards |
| `outputs/stage42_long_research/` | Long-run audits, ablations, replay reports, and source/domain analysis |
| `outputs/stage43_latent_state/` | Protected latent-state and bounded-residual evidence |
| [`research_state.json`](research_state.json) | Machine-readable project state |

Large datasets, generated caches, checkpoints, videos, third-party raw data, and local virtual environments are intentionally left out of Git.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training path is designed for long local runs with single-process loading, checkpoints, heartbeat logs, resume support, and CPU/MPS-safe execution.

Basic test command:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Next Milestone

The next milestone is to make neural world dynamics carry more of the result instead of relying so heavily on protected selection and fallback. That means improving weak source and horizon slices, auditing timing and geometry source by source, and proving through ablations that scene, goal, interaction, and latent-state signals add value under the same no-leakage rules.

M3W is not a finished claim. It is a research path toward a stronger real-world multi-agent world model, with the evidence kept visible as the project moves.
