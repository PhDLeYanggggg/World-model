# Exact Reference Replay: Explicit Execution Amendment

This amendment is registered after a TRAIN-only reproducibility failure, before
any new development inference. It changes one historical-artifact requirement;
it does **not** increase floating tolerances, change the training objective or
relax the scientific utility/risk screen. The original registration and failed
job remain intact and traceable.

## Observed Failure And Diagnosis

Task37814169_0 completed2,000 quadratic updates, then failed bitwise equality
against its historical frozen no-auxiliary checkpoint. Diagnostic37814380
completed0:0 in32seconds on the same AMD EPYC7713 node. It performed5,900
verification updates: original and new direct2,000-step fits, plus resuming
the original100-step pilot control for1,900 additional steps.

The following comparisons are exact for model, optimizer, initialization,
preprocessing, sample stream and random states:

- Original direct versus new direct training.
- New direct training versus the failed task's saved checkpoint.
- Original training resumed from the pilot versus the failed task.
- Original direct versus original pilot-resumed training.

Only historical versus new direct execution differs, and only in model and
optimizer tensors. Its maximum tensor difference is2.428889274597168e-6.
The first unequal recorded monitoring value is atstep200, differing by
5.960464477539063e-8. Inputs, initialization, preprocessing, updates and random
draws remain exact. The new implementation and resume path are therefore
reproduced by the unmodified original trainer. The precise historical
floating-point/kernel cause remains unresolved; CPU portability is not claimed
as a fully proven low-level diagnosis.

Task1 completed36 fits in5m17s; its18 historical quadratic controls match exactly.
Those fits and checkpoints are preserved unchanged.

## Narrow Replacement Check

For the single registered first manifest identity, the new quadratic checkpoint
must exactly match the diagnostic's hash-frozen **original-trainer direct**
checkpoint. Metadata, initialized parameters, input hashes, preprocessing,
budget, sample order and random states must also match the historical control.
No tolerance is introduced. A different identity, changed schema/sampling,
incomplete budget or mismatch with the original-trainer replay still fails.

Every other quadratic head retains the historical bitwise-equality check.
Report the two facts separately:71 historical exact controls and1 exact
original-implementation replay, rather than claiming72 historical reproductions.
The previous historical mismatch remains a recorded reproducibility limitation.

## Resume And Ownership

Resume only shards0/2/3 in a new three-task array. Reuse the failed task's valid
2,000-step quadratic state and100-step deviance pilot; do not discard them or
rerun the36 completed fits. Preserve the earlier job receipts and failed logs
in an owned execution archive. Retire only the old held tasks2/3 and their
now-obsolete pending join after the replacement jobs are recorded. No running
or unrelated task is cancelled.

The new join must verify the completed old task1 and all three resumed tasks,
then all144 checkpoint hashes, fixed2,000 steps and both types of exact control.
Main training remains144 fits at2,000 steps; the5,900 diagnostic updates are
reported separately. Training and future readout retain their original roles,
three seeds,2% budget and fixed thresholds. Independent roles remain closed.

No new trajectory training, deployment promotion, metric/seconds conversion,
Stage5C execution or SMC. This is a reproducibility amendment, not model lift.
