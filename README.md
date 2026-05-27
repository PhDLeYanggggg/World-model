# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on multi-agent world modeling from real scene and trajectory data.

The practical question I am working on is simple to state and hard to solve:

> Given a real top-down scene and the recent motion of every visible agent, can a model predict what the world state is likely to do next, without leaking future information and without becoming unsafe on easy cases?

The current answer is promising but still deliberately bounded. M3W is not a true 3D world model, not a foundation model, and not a metric or seconds-calibrated system yet. The evidence in this repository is mainly for protected raw-frame / dataset-local 2.5D multi-agent trajectory world-state modeling.

## Current Position

The best current model family is a protected M3W-Neural v1 line built around a teacher-floor / safety-floor policy.

In plain terms, the neural model is allowed to improve a strong causal prediction only when a validation-selected safety policy says the switch is likely to help. When the model is uncertain, risky, or likely to hurt an easy case, the system keeps the safer baseline prediction.

That design choice came from the experiments, not from taste. Ungated neural variants often improved one slice of the benchmark while damaging easy cases, proximity behavior, or long-horizon stability. The protected policy is currently part of the method.

## What M3W Currently Does

M3W currently operates as a protected 2.5D multi-agent world-state model. It uses:

- causal history windows
- neighbor and density context
- goal / route prototype features
- source and domain information
- baseline rollout features
- safe-switch and bounded correction policies
- waypoint and endpoint evaluation
- hard/failure and easy-case audits
- no-leakage gates

The model is evaluated against strong causal baselines and earlier selector baselines, not just against weak neural demos.

## Results I Currently Trust

These are the results I am comfortable summarizing at the project level. They are still raw-frame / dataset-local results, not metric or seconds-level claims.

| Line of work | What it contributes | Current interpretation |
| --- | --- | --- |
| Stage26 SDD cost-aware selector | Strong SDD pixel-space deployable baseline | Reliable SDD safety baseline |
| Stage37 external t+50 safe selector | Repaired external t+50 transfer with positive bootstrap evidence | External safety floor |
| M3W-Neural v1 protected dynamics | Neural proposals under Stage37 / teacher-floor protection | Current protected neural candidate |
| Stage42 full-waypoint / group-consistency family | Moves beyond endpoint-only behavior toward all-agent waypoint evidence | Best current world-state evidence |

The short version:

> M3W is currently a protected raw-frame 2.5D multi-agent world-state candidate that improves over strong causal and selector baselines on the current benchmarks while preserving easy-case safety under a conservative floor policy.

That is the claim. I am not claiming more than that.

## What I Am Not Claiming

The following are still blocked or out of scope for current claims:

- true 3D world modeling
- foundation-model scale or generality
- global metric coordinates
- seconds-level horizon claims
- ungated neural deployment
- Stage5C latent generative execution
- SMC readiness
- human-gold labels for self-audited or visual-prior annotations

SDD is treated as pixel-space. External datasets are treated as dataset-local or weak/diagnostic unless their source terms, homography, scale, FPS, and frame stride are verified.

## Why The Repository Has So Many Reports

I keep the failed routes because they are part of the result.

Some important negative findings so far:

- Hard one-hot baseline selection was not safe enough.
- JEPA did not become an independent downstream contribution yet.
- Zero-shot SDD to external transfer failed before domain and horizon repairs.
- Latent distribution alignment alone did not give predictive lift.
- Ordinary residual correction was not deployable.
- Ungated Transformer / Hybrid predictions did not beat the protected floor.
- Scene/goal and neighbor/interaction features are useful in protected settings but are not yet stable standalone main claims.

Those failures shaped the current protected design.

## How To Read This Repo

The root README is the public overview. The detailed research ledger is elsewhere.

| Path | Purpose |
| --- | --- |
| `src/` | data processing, models, evaluation, gates, report builders |
| `configs/` | training and evaluation configs |
| `tests/` | regression tests and evidence checks |
| `outputs/m3w_neural_v1/` | current M3W-Neural v1 summaries and model cards |
| `outputs/stage42_long_research/` | current long-run evidence package, gates, claim guards, ablations |
| `README_RESULTS.md` | internal experiment ledger |
| `research_state.json` | machine-readable project state |

Large local data, fast caches, checkpoints, third-party raw files, videos, and image assets are intentionally not committed.

## Reproducibility Notes

On my Apple Silicon machine I use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Runtime choices that matter:

- `num_workers = 0`
- CPU-safe and MPS-safe paths where applicable
- checkpoint / heartbeat / resume support for longer runs
- no future endpoint input
- no central velocity as official input
- no test endpoints for goal construction

Typical verification:

```bash
.venv-pytorch/bin/python -m pytest tests
```

## Current Research Direction

The next steps are not about making a louder claim. They are about making the claim harder to break:

- broader external top-down validation
- cleaner source and horizon support
- stronger all-agent waypoint evidence
- safer floor relaxation where validation supports it
- better independent contribution from scene, goal, interaction, and neural dynamics modules
- eventual metric/time calibration only when source evidence allows it

Until those are solved, I consider M3W a strong protected 2.5D world-state candidate rather than a finished world model.
