# Training-Time Diagnostic Operations

Run from the repository root with native arm64 `.venv-pytorch/bin/python`.
The runner sets four Torch compute threads, one interop thread and uses no
DataLoader multiprocessing. Original numerical states are immutable inputs.

1. Run the scoped regression tests, including segmented exact replay.
2. Register and commit the fixed protocol and source hashes before the pilot.
3. Run `scripts/run_m3w_european_aux_trajectory.py --phase pilot`.
4. Run the same script with `--phase run`; existing checkpoints resume.
5. Commit `training_freeze.json` before scientific aggregate readout.
6. Run `scripts/report_m3w_european_aux_trajectory.py`.
7. Run the training runner with `--phase verify` to replay every measurement
   from its saved snapshot and compare final states to the original run.
8. Repeat aggregation, confirm byte-identical reports/figure, run scoped tests
   and seal public artifact/source hashes before the final GitHub update.

Private outputs live in
`data/stage_cvpr2027_experiments/european_aux_trajectory_v1/`.
`heartbeat.json` records PID, head, arm and update; `events.jsonl` keeps the
chronological record. The exclusive lock prevents duplicate writers. Atomic
training checkpoints are saved every 200 updates and at measurement points.
Completed snapshot JSON receipts bind model hashes. Full resume checkpoints
retain optimizer, sampler and Torch RNG state. An interrupted uncommitted
measurement is reproduced; a missing earlier measurement after advancement
is an integrity failure, not permission to invent it. At least 10 GiB must
remain free. No raw data or checkpoint is committed to Git.

CREATE was checked read-only before registration; no remote workload was
submitted, interrupted or changed. The experiment runs locally because its
size and existing local assets make that appropriate, not because HPC failed.
Independent data roles remain unopened. Stage5C and SMC remain disabled.
