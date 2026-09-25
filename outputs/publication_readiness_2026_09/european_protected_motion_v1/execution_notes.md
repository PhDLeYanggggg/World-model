# Execution and Reproduction Evidence

The config, model, runner, initial tests and matrix were frozen in commit
`9a889cbed61db26f4d38c6ae0f53f2aeef16bb3c` before the comparative readout.
This completion package changes reporting and adds independent reporting/input
checks, not scientific V1 artifacts or checkpoints.

| Phase | PID | Completion UTC, 2026-09-25 | Evidence source |
|---|---:|---|---|
| Real 100-update pilot | 28110 | 00:25:45 | fresh_run; resumes inside declared budget |
| All 45 damping heads | 28203 | 00:28:22 | fresh_run; 18 ridge, 27 neural, 54,000 updates |
| All 48 policy readouts | 28203 | 00:33:54 | fresh decisions for both candidates |
| Complete metric reconstruction | 28873 | 00:35:54 | cached_verified, no new fitting |
| All 90 checkpoint replays | 28873 | 00:37:09 | fresh inference, 4,096 sampled rows each, exact |

All required training and verification processes exited successfully. Training
was not shortened for speed. The 45 neural-candidate heads came from preceding
verified experiments; they are not included in the new-update count.

## Runtime and Cost

Native arm64 `.venv-pytorch`, Torch 2.12.0, NumPy 2.4.6, CPU compute threads 4,
inter-op 1, DataLoader workers 0. Architecture is checked before Torch import.
No resource probing, DataLoader multiprocessing or NumPy training fallback.
The full process was observed near 8.56 GB RSS. Local resources were adequate;
no CREATE job was submitted or existing simulation job/environment changed.
The exact remote M3W directory remains unverified, not scanned speculatively.

Each new neural head has 22,914 parameters and 2,000 updates. Their summed fitting
loop time is 39.32 seconds, excluding data assembly, identity checks, inference,
solver work and bootstrap. Training plus readout took approximately eight minutes
after launch; verification/replay another three minutes. These are small heads
on frozen trajectory predictions, not an end-to-end world model trained in that
time. The pilot is included in the 54,000-update budget.

## Verification

Detailed local analysis SHA256:
`f3432f5b8ad9ce5071452896638f92fa7f5cf6914c4e357fb51c2434a3224b81`.

- Complete metrics are reconstructed from frozen artifacts; all 48 views match.
- All 90 checkpoints reproduce 4,096 sampled rows each exactly. All nine
  seed/fold groups share 512,000 fitting draws across six neural heads; known
  support, weights, RNG and CV cost scale also match.
- Accounting independently verifies 12 unchanged reference views, 24 unchanged
  neural pointwise views and 144 recording/frame decision receipts. Within each
  candidate, matched independent/unary/joint views have equal actual counts.
- Thirty-eight identity-bound files match hashes. No unknown-label row was
  sampled for new supervision. Future-label poisoning/missingness leaves
  causal decisions unchanged in the additional regression tests.
- Exact-infeasibility pruning is applied to both banks. All current solver
  calls report success; zero old neural decision bits change in 72 receipt views.
- The final 162 tests across 19 scoped files pass in 2.40 seconds, including
  reporting and inference-boundary checks. This is not the full legacy suite
  or independent scientific confirmation.
- The generated contrast figure was visually checked for legibility and overlap.

## Public and Private Artifacts

The public [summary_metrics.json](summary_metrics.json) retains all 48 policies,
all paired contrasts and full-pointwise per-locality metrics, about 1.81 MB.
It binds the detailed local analysis hash. [results.md](results.md) contains all
pointwise/joint/count-matched contrasts; [training_losses.md](training_losses.md)
contains all new fits. The 17.67 MB detailed analysis is reproducible locally
but excluded from this new light-metrics commit, as are PNGs, raw data, private
row arrays, weights and environments. The generated SVG is included.

Private data, events, optimizer/RNG checkpoints and heartbeat remain in
`data/stage_cvpr2027_experiments/european_protected_motion_v1/`.
The [Chinese guide](operation_zh.md) gives exact commands and recovery steps.
Missing private inputs are a reproduction prerequisite, not permission to
invent data or overwrite identity-bound results.

Reserved roles remain closed. No artifact implies independent calibration,
seconds/meters, physical safety, true 3D, foundation, deployment or submission
readiness. Stage5C and SMC remain disabled. Only explicit task paths are committed;
unrelated staged work is preserved.
