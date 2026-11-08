# DroneCrowd Calibration-Only Experiment

**Outcome:** Full inference, scoring and exact replay are complete.
No threshold qualifies. All four switch 362 times across the three
seeds, pass 7/120 recording-seed views and leave the easy risk undefined
in 99 views. Conservative gain against CV is +0.0104 percentage points
(95% source-bootstrap interval [0.0000, +0.0302]). The lower endpoint is
exactly zero. None of the 72 confirmation recordings was opened.

The ungated joint arm passed 69/120 views at the same 67,440 switches as
independent (18/120), shared-scene (22/120), and pair-additive (67/120).
The joint arm's conservative individual ADE gain is +1.315 points over
independent [1.033, 1.636], but only +0.016 over pair-additive [0.010, 0.024].
OLS4 alone is +51.288 points versus CV [46.487, 55.878]; joint is -49.780
points versus OLS4 [-54.471, -44.945], using the observed CV error mass as
the common denominator. OLS4 does not meet the per-view risk rule either.
These are transport and calibration results, not a deployment claim.

[Full decision table](readout/report.md) | [Machine-readable results](readout/readout.json) |
[Exact replay receipt](replay_execution.json)

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

## Verified Outcome Table

After all three phases, regenerate the decision table without reopening
recordings or labels:

~~~sh
.venv-pytorch/bin/python scripts/report_m3w_dronecrowd_calibration.py
~~~

The reporter checks the locked role manifest, the 40-recording
prediction freeze, and the matching score and replay receipts.
It reports each threshold's risk passes, support, interventions and
CV/OLS4 intervals separately, including disqualifying criteria.
It will not generate a partial report or admit the confirmation run.
The full local job completed in 6 h 25 min 36 s on the registered
arm64 CPU configuration. No reduced roster or CREATE run was used.
