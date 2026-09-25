# Frozen Producer Transport Diagnosis

## Material Passport

Code experiment, opened-source development only. Registered before new inference
and comparative readout. No fitting, threshold search, reserved data or deployment.
The starting evidence is the completed nested-calibration study, whose neural
candidate loses all36 direct comparisons to matched protected damping.

## Question

Does changing the cost-target producer from a two-locality OOF predictor to the
four-locality final predictor explain poor intervention decisions? Keep the
head, preprocessing, risk budget, rows, source exclusions and seed fixed. Compare
the two frozen half-fold producers and the final full-fold producer on exactly
the same eight excluded source localities. Retain both halves, every fold and
all three seeds. No outcome selects one of these as a deployed predictor.

This intervention changes producer identity, fitting roster, fitting normalizer
and possibly the train-selected motion floor together. It is not a pure causal
estimate of training-set size. Same-head replacement tests whether producer
transport is a useful repair direction, not whether a differently retrained
cost head could ever work.

## Fixed Comparisons

18 new past-only inference banks from existing two-locality checkpoints. Nine
four-locality banks,54 gain/risk heads,source arrays and splits are cached_verified.
For each fold/seed,compare half0,half1,full4 and damping0.97 under the unchanged
uncalibrated2% all-event and easy-event rules. Calibration maps are not refitted
or reused as certificates for a changed producer. This is72 per-fold policy
views over eight excluded localities each,not72 independent experiments.

Record raw ADE/FDE and controller error;gain/harm estimation bias and MAE;
selected-row versus population risk;oracle opportunity,captured benefit,paid
harm,missed utility and risk-veto opportunity;hard/easy,worst locality,zero-CV
and unknown support. Reconstruct original full4 and damping decisions exactly.
Compare half0/full4 and half1/full4 on identical rows,with3,000 locality bootstrap
resamples. Keep all estimates and intervals,including undefined or negative.

If replacing full4 with its source-excluded half predictors does not consistently
improve the controlled result,do not claim producer-size mismatch caused the
failure. Distinguish candidate quality,score/ranking error and source variability.
Hindsight opportunity is diagnostic,never inference input or a deployable oracle.

## Runtime and Boundaries

Native arm64 Torch,CPU4/inter-op1,workers0,batch128. Pilot4,096 rows to measure
inference cost;then complete all3,827,628 candidate-row predictions if resources
allow. Save complete per-producer banks atomically;resume completed verified
banks,not an arbitrary partial prediction. Replay4,096 rows per checkpoint.
No new model training is represented as occurring in this diagnostic.

12 opened source localities only. Raw detector tracks,image pixels,obs8/pred12
rawstride12,not historicalt50,seconds,metric,human-gold,online sensor certification,
physical safety,true3D or foundation. No joint-controller contribution or
independent confirmation is established here. No Stage5C or SMC.
