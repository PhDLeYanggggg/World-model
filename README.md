# M3W

M3W is my research project on real-world multimodal, multi-agent world modeling.

The question I am working on is:

> Given a real top-down scene and the recent motion of every visible agent, can a model predict the next world state without leaking future information and without becoming unsafe on easy cases?

I am interested in the practical version of that problem, not a polished demo. The repository therefore keeps both the working route and the failed routes: strong causal baselines, selectors, neural dynamics heads, safety floors, no-leakage audits, and the reports that explain why some ideas were rejected.

## Where The Project Stands

M3W is currently a protected 2.5D multi-agent world-state modeling system. It is not a true 3D world model, not a foundation model, and not a metric or seconds-calibrated system yet.

The current evidence is raw-frame / dataset-local:

- SDD results are pixel-space.
- External top-down pedestrian results are dataset-local unless source geometry is verified.
- t+50 and t+100 are raw-frame horizons, not seconds-level claims.
- self-audited and visual-prior labels are not human-gold labels.
- Stage5C latent generative execution has not been run.
- SMC is not enabled.

That boundary matters. I would rather have a narrower claim that survives scrutiny than a louder one that collapses under audit.

## What Works So Far

The strongest useful pattern so far is not “let a neural model overwrite everything.” It is a protected policy:

1. start from a strong causal baseline;
2. use causal history, neighbor context, route / goal prototypes, source/domain context, and learned risk signals;
3. switch only when validation evidence says the change is likely to help;
4. fall back when the case looks easy, uncertain, or likely to be harmed.

The best current evidence comes from three parts of the project:

- an SDD pixel-space cost-aware selector that is still the SDD safety baseline;
- an external t+50 repair that turned earlier failed transfer into a deployable raw-frame selector candidate;
- protected full-waypoint / group-consistency policies that move the work beyond endpoint-only behavior.

The short version:

> M3W is currently a protected raw-frame 2.5D multi-agent world-state candidate. It has useful evidence on the current benchmarks, but it is still deliberately bounded and safety-floored.

## What Failed And Why It Stayed In The Repo

Several routes did not become main claims:

- hard one-hot baseline classification over-switched and hurt easy cases;
- JEPA avoided collapse but did not yet produce reliable downstream lift;
- zero-shot SDD-to-external transfer failed before domain and horizon repair;
- latent distribution alignment reduced distance but did not reliably improve prediction;
- ordinary residual correction was not safe enough to deploy;
- ungated Transformer / Hybrid dynamics did not beat the protected floor;
- scene, goal, and interaction features help inside guarded policies but are not yet standalone main contributions.

Those negative results are part of the work. They shaped the current safety-first design.

## How To Read The Repository

The root README is the public overview. The detailed experiment ledger is intentionally separate.

| Path | What it is for |
| --- | --- |
| `src/` | data processing, models, evaluators, gates, report builders |
| `configs/` | training and evaluation configs |
| `tests/` | regression tests and evidence checks |
| `outputs/m3w_neural_v1/` | current M3W-Neural v1 summaries and model cards |
| `outputs/stage42_long_research/` | long-run evidence, gates, claim guards, ablations |
| `README_RESULTS.md` | internal experiment ledger |
| `research_state.json` | machine-readable project state |

Large local data, caches, checkpoints, third-party raw files, videos, and image assets are not committed.

## Reproducibility

On my Apple Silicon machine I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Important runtime choices:

- `num_workers = 0`
- CPU-safe and MPS-safe paths where applicable
- checkpoint / heartbeat / resume support for long runs
- no future endpoint input
- no central velocity as official input
- no test endpoints for goal construction

Typical verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Current Direction

The next step is to make the claim harder to break:

- broaden external top-down validation;
- strengthen all-agent waypoint evidence;
- keep source / horizon / scene breakdowns visible;
- improve neural dynamics only where it beats the protected floor;
- continue separating deployable results from diagnostic ones;
- attempt metric and time calibration only when the source evidence supports it.

Until those pieces are stronger, I describe M3W as a protected 2.5D world-state candidate, not a finished world model.
