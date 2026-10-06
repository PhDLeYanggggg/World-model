# Real TRAIN Pilot: Exact Execution, No Scientific Lift Claim

Result source: **fresh_run**. CREATE job37813826 completed with exit0:0
in59seconds on2026-10-06. This is a real source-TRAIN cost-head pilot, not a
new trajectory forecaster or a development/independent evaluation.

The preselected first manifest identity is locality008, head seed17. Both
arms ran100 updates. Two additional100-update control/recovery replays bring
the actual optimizer update count to400; they are not400 distinct training
examples or four independent experiments. The pilot pair will resume to the
registered2,000 steps in full training.

## Execution Checks

- Original quadratic model, optimizer and sampling state replay: exact.
- Interrupted50+50 versus uninterrupted100-step deviance replay: exact.
- Paired initialization, preprocessing and sampled queries: identical.
- Finite real-TRAIN decoder gradients: verified for both objectives.
- Peak Python RSS:1,291,505,664bytes, below the14GiB admission limit.
- Slurm batch peak RSS:1,847,400KiB; this is a separate process-accounting measure.
- CPU threads4, interop1, DataLoader workers0; Torch2.12.0+cpu, NumPy2.4.6.
- Estimated per-shard runtime including the fixed1.5 multiplier:654.21seconds.
  This first-identity extrapolation is not a measured full-run guarantee.
- Combined parent/new checkpoint payload:49,537,227bytes at pilot completion,
  below256MiB. The10GiB storage reserve remains enforced.

## Monitoring Losses

| Arm | Initial own objective | Final own objective | Final original quadratic objective |
| --- | ---: | ---: | ---: |
| quadratic | 0.9908183813 | 0.9968657494 | 0.9968657494 |
| easy_deviance | 0.9135140181 | 0.9200503826 | 0.9906745553 |

Both own-objective monitoring losses increased slightly. The objectives have
different definitions and their absolute values must not be ranked against
each other. The tiny change in the shared quadratic monitor is not evidence
of downstream superiority. Admission was prespecified as execution safety and
exact recovery, not pilot performance. No hyperparameters were selected here.

The fixed128-query gradient monitor includes1,442 rows and83 positive easy-harm
rows. At the quadratic checkpoint, easy-channel decoder gradient norms are
0.02983 under quadratic loss and0.22753 under deviance. This measures different
loss geometry, not calibrated risk or proof of a successful repair.

## Full Execution

Array37814169 and after-success verification job37814170 were submitted and
released after successful pilot accounting. Four disjoint shards cover all
72 identities and144 paired fits. Each fit retains2,000 updates; every100
updates saves recovery state. Each compute task has a12-hour limit. Full
completion requires all144 final artifact hashes, matched sampling, and72 exact
original quadratic controls. Submitted is not completed.

All final heads must freeze before the fixed source-development readout.
The original forest, additive, positive-harm and cost controls remain visible.
Independent calibration and confirmation stay closed. No checkpoint or
threshold selection, new forecast, deployment promotion, Stage5C or SMC.

Image-local pixel coordinates, raw annotation frames, detector-silver labels:
no metric, seconds-level, true3D, foundation or physical-safety claim.

Evidence: [pilot.json](pilot.json), [pilot_collection.json](pilot_collection.json),
[train_submission.json](train_submission.json), [protocol.md](protocol.md).
