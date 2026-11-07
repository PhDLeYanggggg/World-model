# DroneCrowd Calibration-Only Experiment

**Launch status:** Past-only inference is running on the full approved
calibration role. No calibration score is available at launch. The
runner cannot access any of the 72 confirmation recordings.

The [registered protocol](protocol.md) fixes the exact three-seed
forecaster bank, causal comparators, development controller, four
normalized gain thresholds and selection rule. The original 2% and
reference-support requirements are unchanged. A threshold that switches
zero times or leaves a required risk undefined cannot win.

The [real-model smoke](smoke_execution.json) ran 16 queries and 3,085
history-complete target instances from two calibration recordings. It
exercised all nine checkpoint/seed combinations using native arm64,
four Torch threads and zero DataLoader workers. It took 4.59 s for 261
rows and 46.19 s for 2,824 rows; linear extrapolation gives about 8 to 9
hours for the full 1,757,584-row inference. That is an estimate, not a
completion claim. Scoring and numerical replay are separate phases.

## Local Recovery

The supervisor starts the registered `infer`, `score`, and `replay`
phases in order. The score phase refuses to read labels unless all
40 forecast artifacts and their exact roster are frozen. Runtime files
stay under the ignored calibration directory:

```sh
base=data/stage_cvpr2027_experiments/dronecrowd_calibration_readout_v1
jq . "$base/job_state.json"
tail -n 20 "$base/calibration.log"
jq . "$base/heartbeat.json"
```

Check the actual supervisor and worker PIDs with `ps`. Do not infer
failure from a quiet log during a dense recording. Once the original
job has exited, resume with:

```sh
.venv-pytorch/bin/python scripts/launch_m3w_dronecrowd_calibration.py start
```

The run resumes verified recording checkpoints and does not overwrite
an existing result. macOS sleep prevention is tied to the supervisor.
No CREATE job was needed. Neither calibration nor replay constitutes
independent confirmation. No deployment, Stage5C execution or SMC.
