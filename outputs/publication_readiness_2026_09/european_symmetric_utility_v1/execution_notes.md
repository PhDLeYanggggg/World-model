# Execution Evidence

Registration/code/config commit: `38fdb034587ba3a66266b108ef67ac3d6604c931`,
pushed before the comparative outcome readout. Frozen scientific files were not
edited after the pilot. Additional reporting code does not change any policy.

| Phase | PID | Completed UTC, 2026-09-25 | Source |
|---|---:|---|---|
| 100-update real pilot | 32573 | 01:58:08 | fresh_run, inside fixed budget |
| All fits and 48 policy readouts | 32621 | 02:06:24 | fresh_run |
| Complete metric reconstruction | 33292 | 02:07:57 | cached_verified arithmetic, no fitting |
| All 18 checkpoint replays | 33292 | 02:08:42 | fresh inference, exact |

All required training and verification processes exited zero. Native arm64,
Torch2.12.0, NumPy2.4.6, CPU threads4/inter-op1, workers0. No resource probing,
DataLoader multiprocessing or NumPy training fallback. Process RSS was observed
around 6.1GB. Approximately 29GiB free disk was available during the run.
No CREATE job was needed or submitted; simulation tasks were not touched.

Each small utility head has 22,914 parameters. Summed fitting-loop time is
27.82 seconds, excluding data assembly, lineage/hash checks, inference, solver
work and bootstrap. Training plus readout took about eight minutes; verification
and replay about two minutes. This does not mean an end-to-end world model was
trained in minutes: all trajectory and risk models were frozen inputs.
The full 36,000-update budget and complete source population were retained.

Detailed analysis SHA256:
`f664e03e0d5351f029af0964627106d1496f1c0f4607e5b5f20ddee018467bab`.

Light summary SHA256:
`b9097ea9c01abadd393ad697b8127f1a8f2ff03e1b72ca68be9e69fdc9e358f7`.

The local detailed analysis is 16,535,701 bytes and ignored. Public summary is
1,659,094 bytes and retains every registered view, paired contrast and full
per-locality metric. Private checkpoints, row arrays, events and RNG states
remain under `data/stage_cvpr2027_experiments/european_symmetric_utility_v1/`.
Neither those artifacts nor raw data, third-party images or environments are
included in Git. The scientific SVG is public; the rendered QA PNG is ignored.

Verification checks every new head's 4,096-row replay, every new-versus-old
sampler/preprocessor pair, and all 144 causal decision receipts. All 90 frozen
control-head artifact records match the prior verified study. Unknown-label
supervised draws are zero. Actual within-candidate matched counts and original
predicted-risk budgets are checked from arrays, not just copied report fields.

179 tests across 23 scoped files pass in 2.68 seconds. The preflight had 24
focused passes. Full legacy-suite execution is not claimed. The generated
all-view figure was rendered and inspected for clipping/overlap.

One joint exact-count solver result fell back safely and is explicitly reported,
not counted as a successful exact solve. It does not invalidate checkpoint or
metric reproducibility; it does prevent claiming every optimization succeeded.
No new result is independently calibrated, deployment-promoted or paper-ready.
