# Execution Record

Registration commit:b6ea4f8e. Source-held readout must follow a separate committed
prediction freeze. No independent selection,calibration or confirmation access.
Parent local and GitHub HEAD were2340fe113a5e76c68af4be7a62d30684d6313c85.
All18 public artifacts and23 source bindings of the preceding trajectory study
matched its verification seal before edits. Existing staged work was preserved.

## Runtime and CREATE
Native arm64 .venv-pytorch,Python3.11.1,NumPy2.4.6,PyTorch2.12.0,
CPU4,interop1,workers0. Free disk14.28GiB at preflight;
runner retains10GiB. Exclusive lock,atomic checkpoint and heartbeat every200
updates. No resource-probing or multiprocessing path. Runtime adequacy is judged
by the real200-update pilot,not import success.

CREATE queue query on2026-09-27 succeeded read-only (receipt SHA256
36223426201905181500855e3a289e2a9f6440333890619aa373a8ef807de50d).
Jobs37542192,37542492,37542323 were pending priority/dependency. Their contents
and outputs were not inspected,and they are not claimed as M3W experiments.
No remote job was submitted,modified or stopped. Local data avoids transfer.

## Reproduction
From repository root,using the native environment:

```bash
.venv-pytorch/bin/python scripts/run_m3w_european_aux_prior.py --phase pilot
.venv-pytorch/bin/python scripts/run_m3w_european_aux_prior.py --phase train --resume
# Commit prediction_freeze.json before the next step.
.venv-pytorch/bin/python scripts/run_m3w_european_aux_prior.py --phase evaluate
.venv-pytorch/bin/python scripts/report_m3w_european_aux_prior.py
.venv-pytorch/bin/python scripts/run_m3w_european_aux_prior.py --phase verify_training
.venv-pytorch/bin/python scripts/run_m3w_european_aux_prior.py --phase verify_eval
```

Register on a clean new experiment directory only. Immutable completed artifacts
must not be overwritten;the existing completed run is replayed with verification
phases. Checkpoints,aligned predictions and logs live under the ignored
data/stage_cvpr2027_experiments/european_aux_prior_v1 directory. Public aggregates
cannot recreate private source data. Git contains no raw rows or weights.

Scoped preflight:16 tests passed across3 files,including exact legacy numerical
identity for3 arms,intercept isolation,matched true/shuffled priors,resume and
gate guards. This is not the full historical test suite or empirical efficacy.

## Completed Training
Pilot PID50870 completed200 updates and resumed in training PID51197. The full
run completed288 heads/576000 updates,with zero unknown-label draws and288
intercept-isolation/matched-sampler checks. Pure recorded fitting time was
587.67093 seconds;run-phase wall time933.50591 seconds excludes ancestry
preflight and reuses the pilot. No training was skipped or reduced.
Memory snapshots stayed below13GiB and free disk stayed above10GiB.
Prediction-freeze commit9eb1ca68 preceded the source-held readout (PID52832).

The detailed9.2MiB readout.json is kept locally and excluded through this
repository's local exclude file. The verification receipt hashes it separately
as local_detailed_metrics. Git receives aggregate contrasts,all assignment
intervals,figures,receipts and reports,not this repetitive detailed metric file.
