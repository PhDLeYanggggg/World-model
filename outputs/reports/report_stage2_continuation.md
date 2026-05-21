# Stage 2 Continuation Notes

## What Was Changed

This continuation targets the three failures from the previous quick run:

1. `coverage@64 = 0` because particles mostly shared the same goal and only differed by local noise.
2. terminal clusters had weak semantics because `collision_risk` absorbed many otherwise meaningful rollouts.
3. quick evaluation was biased because it selected smallest-agent test episodes, which can repeat the same event mode.

Implemented changes:

- `configs/stage2.yaml`
  - `quick_eval_episodes` increased from `3` to `4`.
  - Added latent-goal SMC knobs:
    - `latent_goal_keep_probability`
    - `latent_goal_exit_probability`
    - `latent_goal_noise_m`
    - `latent_goal_rejuvenation_probability`

- `src/data/synthetic_dataset.py`
  - Synthetic split generation now cycles through scene templates instead of fully random scene choice.
  - This makes the quick train/val/test split less event-narrow.

- `src/data/synthetic_physical_crowd.py`
  - `smooth_passage` classification now preserves true open-space scenes when they are not jammed or dense.

- `src/training/evaluate_stage2.py`
  - Quick evaluation now uses an event-balanced selector instead of smallest-agent-only sorting.
  - Report metadata includes selected event labels and explicitly states the event-balanced selection policy.

- `src/inference/smc_rollout.py`
  - SMC particles now initialize with latent goal hypotheses.
  - Particle goals can be:
    - kept near the observed/current goal,
    - sampled from scene exits using velocity-heading bias,
    - extrapolated along current motion.
  - After resampling or long rollout intervals, goals can be rejuvenated.
  - This turns SMC branches into different intention hypotheses, not just noisy copies.

- `src/inference/cluster_futures.py`
  - Semantic rules now classify stall/jam/detour/split/high-density/smooth modes before fallback collision risk.
  - `collision_risk` is no longer allowed to swallow most otherwise meaningful trajectories unless gaps are truly severe.

## Verification

Static verification passed:

```text
python -m py_compile src/data/synthetic_dataset.py src/data/synthetic_physical_crowd.py src/inference/smc_rollout.py src/inference/cluster_futures.py src/training/evaluate_stage2.py run_stage2_demo.py
```

## Current Runtime Blocker

Re-running `python run_stage2_demo.py` currently triggers:

```text
OMP: Error #179: Function Can't open SHM failed
OMP: System error #0: Undefined error: 0
```

Two attempted runs entered uninterruptible `U` state:

```text
95790 UEs  python run_stage2_demo.py
96430 UEs  python run_stage2_demo.py
```

They did not respond to normal `kill` or `kill -9`, which means this is below Python-level model logic. It is an OpenMP/shared-memory runtime issue in the local macOS/Conda/PyTorch stack. The source changes are compiled, but the updated metrics have not yet been regenerated.

## Expected Next Check After Runtime Clears

Run:

```bash
python run_stage2_demo.py
```

Then inspect:

- `outputs/reports/metrics_stage2.json`
- `outputs/reports/report_stage2.md`
- `outputs/figures/stage2/terminal_clusters_semantic.png`

The specific questions to check are:

1. Does latent-goal SMC improve best-of-N t+100 FDE?
2. Does `coverage@64` become non-zero or at least improve under a looser distance threshold if added?
3. Do terminal clusters include more than `collision_risk / physically_invalid / uncertain_multimodal`?
4. Does event-balanced quick eval expose corridor, detour, split, and smooth modes more fairly?

Do not claim improvement until the updated run finishes.
