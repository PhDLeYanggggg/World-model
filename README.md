# M3W

M3W is my research project on real-world multimodal multi-agent world modeling.

The question I care about is simple to state and hard to make honest:

> Given a scene and the recent motion of all visible agents, can a model predict what happens next without using future information, and without making the easy cases worse?

Most of the work in this repository is about earning that answer carefully. That means converting messy real trajectory data, building strong causal baselines, checking leakage, keeping failed ideas in the record, and only promoting a model when it beats the protected fallback under the right gates.

## Where The Project Stands

M3W is currently a protected 2.5D multi-agent world-state system. It is not a finished 3D world model and it is not a foundation model.

The strongest deployable pieces today are guarded selector policies. They start from simple causal motion baselines and switch only when there is evidence that switching is useful and safe. This has worked better than trying to replace the baseline with an unconstrained neural model.

The current results should be read with these boundaries:

- SDD is evaluated in pixel space.
- External trajectory data is dataset-local unless geometry has been separately verified.
- `t+50` and `t+100` are raw-frame horizons, not seconds.
- Scene/goal labels that are inferred or self-audited are not human gold labels.
- Stage5C latent generative rollout is not executed.
- SMC is not enabled.

Those limits matter. I would rather have a smaller claim that survives audit than a larger claim that depends on hidden leakage or loose wording.

## What Has Worked So Far

The most reliable direction has been cost-aware, fallback-protected selection:

- a strong SDD selector that improves the pixel-space benchmark while preserving easy cases;
- an external `t+50` repair that turned failed transfer into a deployable raw-frame selector candidate;
- history-window and source-aware policies that passed bootstrap and replay checks;
- no-leakage audits around future endpoints, central velocity, and test-built goals;
- full-waypoint and long-horizon experiments that are only promoted when they survive the safety floor.

In practical terms, M3W is useful right now as a guarded trajectory/world-state research system. The longer-term target is a stronger multimodal agent-scene world model with real dynamics contribution beyond selector logic.

## What Has Not Worked

Some routes were useful because they failed clearly:

- hard one-label baseline classification switched too aggressively and hurt easy cases;
- JEPA representations avoided collapse, but have not yet given reliable downstream lift;
- zero-shot SDD-to-external transfer failed before domain, horizon, and history repair;
- latent distribution alignment reduced distance without consistently improving prediction;
- ordinary residual correction was not safe enough to deploy;
- ungated Transformer and Hybrid dynamics did not beat the protected floor.

I keep these results visible because they explain why the current system is conservative.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `src/` | data conversion, models, evaluators, gates, report builders |
| `configs/` | training and evaluation configs |
| `tests/` | regression tests and evidence checks |
| `outputs/m3w_neural_v1/` | current model summaries and cards |
| `outputs/stage42_long_research/` | long-run reports, ablations, guards, and evidence tables |
| `README_RESULTS.md` | detailed experiment ledger |
| `research_state.json` | machine-readable project state |

Large local data, caches, checkpoints, videos, images, and third-party raw files are intentionally not committed.

## Running Locally

On Apple Silicon I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

The training and evaluation code is set up to avoid the old x86_64 Conda + Intel OpenMP path:

- `num_workers = 0`;
- CPU-safe and MPS-safe paths where applicable;
- checkpoint, heartbeat, and resume support for long runs.

Typical verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## What I Am Pushing Toward

The next meaningful progress is stronger evidence, not louder claims:

- broader external top-down validation;
- better all-agent waypoint evidence;
- clearer source, horizon, scene, and agent-type breakdowns;
- neural dynamics only where they beat the protected floor;
- metric or seconds-level claims only after dataset geometry and timing audits support them.

M3W is the path toward a real multimodal agent-scene world model. This repo shows that path as it currently is: promising, guarded, unfinished, and intentionally careful.
