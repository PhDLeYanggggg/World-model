# M3W

**Real-World Multimodal Agent-Scene World Model**

M3W is my long-running research project on real-world multi-agent world modeling from top-down pedestrian and agent-scene data.

The question behind the project is:

> Can a model use only past motion, scene context, and local interactions to predict future multi-agent behavior better than strong causal baselines, while staying safe on easy cases and avoiding leakage?

I care about this because many trajectory systems look good only when the benchmark is forgiving: weak baselines, future endpoint hints, optimistic geometry, or test-set information can all make a model seem stronger than it is. M3W is built under stricter rules. At inference time the model only gets past information. If a learned component cannot beat the protected fallback under validation-selected rules, it stays diagnostic.

## Current Position

M3W is currently a protected, dataset-local 2.5D multi-agent world-state model. It is not a true 3D world model, and it is not a foundation model.

The strongest evidence so far comes from conservative policies that combine causal history, interaction features, scene or goal context when it is legally available from training data, and a strong fallback baseline. These policies have improved hard and longer-horizon raw-frame slices while keeping easy-case degradation small.

The neural and latent-state work is active, but I treat it carefully. A Transformer, JEPA encoder, hybrid head, residual correction, or latent policy only matters if it improves the protected policy without breaking the safety constraints. Otherwise it remains an experiment, not the deployed model.

## What M3W Uses

The project has explored several families of signals:

- past-only trajectory history;
- speed, acceleration, heading, curvature, stop/go, and drift cues;
- local density, nearest-neighbor, and time-to-collision features;
- train-only goal prototypes and route priors;
- strong physical baselines such as causal velocity and damped velocity;
- failure, gain, harm, and switchability predictors;
- protected latent-state and bounded-residual policies;
- conservative fallback rules that choose the causal floor when learned switching is uncertain.

The design principle is simple: learning is useful only when it knows when to defer.

## What Has Worked

The most reliable progress has come from cost-aware selection rather than hard class prediction. Earlier selectors that tried to predict the single oracle-best baseline switched too aggressively and damaged easy cases. Later policies predict risk, expected gain, and harm, then switch only when the validation-selected safety rule allows it.

The external transfer story also changed over time. Raw SDD-to-external zero-shot transfer failed. Coordinate repair, causal history windows, scene-agnostic goal prototypes, and horizon-specific policies were needed before external raw-frame `t+50` transfer became useful. That result is still dataset-local and raw-frame, but it is real evidence that the system can learn something beyond an SDD-only policy.

## What Has Not Worked Yet

Some directions are still open or negative:

- JEPA-style representation learning has avoided collapse but has not consistently produced downstream lift.
- Latent distribution alignment can reduce domain distance without improving prediction.
- Ordinary residual correction is unsafe unless it is tightly bounded and protected by fallback.
- Unprotected neural dynamics has not replaced the Stage37-style protected policy.
- Raw-frame `t+100` remains mostly diagnostic and is not a seconds-level claim.
- ETH/TrajNet-style external coverage still needs stronger held-out validation before I would call the model broadly cross-domain.

I keep these failures visible because they are part of the research, not cleanup.

## Claim Boundaries

These are the boundaries I use when reading the results:

- SDD is a pixel-space benchmark unless a source-specific calibration proves otherwise.
- External top-down pedestrian data is dataset-local/raw-frame unless timing, homography, and scale are verified.
- Raw-frame `t+50` and `t+100` are not seconds-level horizons.
- Self-audited or inferred scene labels are not human gold labels.
- Future endpoints, central velocity, and test endpoints are not inference inputs.
- Latent-generative Stage5C has not been executed.
- SMC is not enabled.

The current project is a serious step toward multimodal multi-agent world modeling, but the honest description is still: protected dataset-local 2.5D world-state modeling.

## Repository Map

| Path | Purpose |
| --- | --- |
| `README_RESULTS.md` | Main experiment ledger and high-level results |
| `README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md` | Chinese route-by-route summary of successes, failures, and causes |
| `outputs/m3w_neural_v1/` | Neural model reports and model cards |
| `outputs/stage42_long_research/` | Long-run audits, ablations, replay reports, and source/domain analysis |
| `outputs/stage43_latent_state/` | Protected latent-state and bounded-residual evidence |
| `research_state.json` | Machine-readable project state |

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
