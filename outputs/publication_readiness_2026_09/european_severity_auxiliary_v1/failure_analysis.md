# Severity Auxiliary Failure Analysis

## Scope

Completed fresh registered fitting/readout on already exposed source
development. No independent confirmation or new trajectory evaluation.
The primary and tail/coverage gates fail. Future errors only supervised
loss/evaluation; causal input and frozen denominator definitions unchanged.

## Evidence-Supported Taxonomy

1. Not an absent training run:144 genuine Torch fits completed their2,000
   updates; no unknown labels were sampled. Losses decrease and checkpoints
   exist. Runtime or NumPy fallback does not explain this result.
2. Support is numerically present, but concentrated. Every full fitting view
   has easy harm in3 localities and at least165 recording-agent tracks.
   Median easy-harm mass concentration is68.56 tracks /7.79 recordings,
   versus15.71 /4.97 for motion-only. These are not independent sample sizes.
   Counts establish neither adequate power nor predictable conditional harm.
3. Target alignment alone is insufficient. H-weighted BCE has the desired
   population fraction as its optimum, yet0/6 full easy-harm MSE intervals
   improve vs original. A correct population loss identity is not proof of
   finite-sample estimation or transport with these features.
4. Fitting-to-held transport is inconsistent. Versus original,42/72 full
   views improve fitting MSE,25 improve fitting only,34 improve held and17
   improve both. Motion-only:47 fitting,35 fitting-only,28 held,12 both.
   Both fit failures and transport failures exist; do not call everything
   overfitting. Against ordinary auxiliary:50 fitting/30 fitting-only/27 held
   in full,54/38/26 in motion-only.
5. Ranking and magnitude separate. Four full positive-disagreement AUROC
   intervals improve vs original, while easy-harm MSE has0 positives and
   top10 harm capture has1 negative interval. Improved ranking cannot be
   presented as calibrated harm or successful deployment.
6. Source heterogeneity remains substantial. For producer0/controller1 the
   full MSE point is -0.5868%, CI[-1.0018,-0.1717]. For producer2/controller1
   it is -20.1401%, CI[-48.7137,+0.7918]; the three-seed locality point at
   eu-locality-110 is -65.1841%. Producer0/controller2 has -57.2432% at
   eu-locality-067. These post-readout examples locate error, not permission
   to exclude localities or change the gate. The full table retains all.
7. Tail/coverage tradeoffs are concrete. Full producer2/controller0 top10
   capture vs original is -7.4203pp, CI[-13.0134,-1.8272]. Full producer1/
   controller0 coverage-log-error reduction is -0.04175,
   CI[-0.06862,-0.01489]. No sign or adverse subgroup is hidden.

## Not Yet Established

Concentrated label weights may increase gradient variability, and missing
causal context or locality shift may limit severity estimation. This run
does not causally distinguish those explanations. Nor does it prove that
more epochs, larger models, stronger regularization or clipping will help.
The parent frozen-batch analysis did not support broad total-cost gradient
conflict. Plain harm-only and ordinary auxiliary failures remain evidence;
neither should be relabeled as an untested solution.

## Next Discriminating Work

Use frozen models and existing development rows. Attribute signed excess
squared error to recordings and tracks, preserving both positive and negative
contributions. Compare concentration with fitting-only H-weight mass and
shared-gradient contributions, and assess whether harmful held inputs lack
fitting feature support. Do not collapse recordings/windows into independent
replicates, silently remove difficult rows, or use reserved roles.

If concentrated fitting influence is supported, preregister one controlled
training-stability repair with the original target/strong control retained.
If errors are diffuse or outside fitting support, prioritize causal context
and independently sourced scene support instead. These are conditional next
actions, not results from this round. No threshold rescue or deployment.

Detector-derived pixels and native annotation steps only. No human-gold,
metric/seconds, physical-safety, true3D or foundation claim. Stage5C/SMC off.
