# Candidate-Invariant Reference Cost Ablation

## Material Passport

Prospective bounded repair,29October2026. A completed36-state TRAIN diagnosis
finds that the true reference R/ER targets are candidate-invariant but their
predictions vary substantially between candidate actions. This is structural
incoherence, not proof that correcting it will improve utility or safety.

## One Changed Factor

Keep the five-moment architecture and25,157parameters,initial weights,physical-
fitting normalizers,batch draws,loss,optimizer and2,000updates unchanged.
Compute R/ER once from a fixed causal floor feature token with a zero action
code. Evaluate the same network on candidate feature/action tokens for B/H/EH.
Repeat the shared R/ER over candidates. Enforce the same envelope and positivity
bounds,including benefit<=reference. No additional trainable branch is added.

The floor token is built from the existing past geometry,CV rollout,floor
rollout and causal floor switch. It adds no new outcome,neural feature,agent
identity,scene goal or test statistic. Its normalizer is the original physical-
fitting normalizer. Candidate-specific rollout features no longer influence
the reference estimate; this can lose predictive information and must be tested.
Lazy sampled pairing avoids materializing a doubled full feature tensor.

## Matched Experiment

Six frozen contexts x three controller seeds x physical-only,mixed,neural-only
banks =54fits/108,000primary updates. The forecaster stays seed43 and frozen.
Require an all-bank real10+10/direct20resume pilot and a full2,000update
disabled-sharing neutral replay on the first mixed seed17context. Original
scientific state must match except declared registration/runtime identities.
Every primary fit must match its old initial weights,preprocess,row draws,
sampler/Torch RNG and parameter count. Count the extra shared forward call as
real compute; it is not a zero-cost change.

Freeze all54endpoints before readout. Keep the preceding pointwise2%decision
rule and all fixed harm1/2/4controls. Compare against the unshared parent and
physical-only bank with the same full-policy/utility-only matched readout.
Report all seeds,recordings,known gains,completion bounds,risk support and
unknown selected rows. Sharing references is an implementation invariant,
not a safety gate. No favorable cutoff or best-seed selection after readout.

## Run and Claim Boundaries

Native arm64Torch CPU4/interop1/workers0,checkpoints100,heartbeat50,10GiB reserve,
12h healthy-runtime limit. Checkpoint/resume rather than shrink the roster.
Data remain obs8/pred12/rawstride12 SOURCE development,not historicalt+50 or
independent confirmation. Independent selection/calibration/confirmation stay
closed. Shared heads and cost deferral are not claimed as novel by themselves.
No new trajectory producer,scene-joint optimization,metric/seconds,true3D,
foundation,Stage5C orSMC claim/execution. Deployment changes only after original
required gates pass; neither coherence nor average gain alone is sufficient.
