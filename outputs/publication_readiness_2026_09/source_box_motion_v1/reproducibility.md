# Reproduction

Run from the repository with nativearm64 `.venv-pytorch`. Data and the bound
OpenCV runtime must already exist locally. No source data or checkpoints are inGit.

```sh
.venv-pytorch/bin/python scripts/build_m3w_source_box_motion.py
.venv-pytorch/bin/python scripts/run_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json
.venv-pytorch/bin/python scripts/run_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json --replay
.venv-pytorch/bin/python scripts/analyze_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json
.venv-pytorch/bin/python scripts/verify_m3w_source_box_motion.py --registration configs/m3w_source_box_motion_v1.json
.venv-pytorch/bin/python scripts/report_m3w_source_box_motion.py
```

Repeating the training command verifies receipts and resumes incomplete fixed
trials. Never delete an active checkpoint or start duplicate workers. Atomic
checkpoints every200updates. Private training.log/fit_heartbeat.json record PID,
step and elapsed fit time. Initial100-update pilot is part of the fixed budget.
Interrupted-resume equality is tested on the model; actual pilot is recovered.

Registration: e656507cf789160047444de879acde3b4e0091f423c36e9cea1842bcbd96d07a
Analysis: 2884fc1fe37bccb44d029821744b3ab4e8f9158e185bfcbb2b6a4739644f9cd7
Verification: 22bb4f1c0706d5a474c9e46205fda604fec0ab669308736aff1776fdca108095

Full legacy test suite is not rerun because some integrations overwrite historical
reports. Scoped tests and real artifact checks are reported separately.
