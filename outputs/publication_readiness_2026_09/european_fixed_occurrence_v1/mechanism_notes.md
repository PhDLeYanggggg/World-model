# Mechanism Boundaries Before the New Readout

This note follows completed fitting but precedes the held-development readout.
It adds no model, threshold, score selection or changed gate.

The easy-risk score is `q = p * (h - 0.02 * r)`, where `p` is easy occurrence,
`h` is conditional positive harm and `r` is conditional reference cost.
For a strictly positive `p`, changing only `p` cannot change the sign of an
individual score with fixed `h,r`. It can change magnitudes, hence query-level
joint allocation. Numerical zero probabilities are an exception. Freezing `p`
also changes cost-branch training through the marginal-risk loss and gradient
clipping. Therefore this paired training experiment is not a pure inference-only
probability substitution. It isolates branch trainability under the specified
optimizer and loss, with an identical split architecture in both arms.

The fitting result is negative for the primary risk loss: fixed0.002507829 versus
trainable0.002378599, only2/108 better groups. Fixed occurrence does not improve
Brier versus continued training; its unchanged probabilities have Brier0.111875
versus0.096806. The fixed sign screen admits more useful switches but more harm.
No fitting quantity selects a checkpoint or modifies the subsequent comparison.

There is also a pre-existing support issue. The original raw independent control
has16 completely abstaining dependent views, verified before this experiment.
The new common matched anchor is a subset of that control, so those views cannot
acquire a nonzero matched count merely by freezing a probability branch. Their
selected-risk ratio remains undefined and cannot be called zero risk. The prior
experiment's19-view lower bound included its other arm and is not copied here.
The original all-view risk screen stays unchanged even though its defined-risk
condition has this structural obstruction. This study can test a developmental
mechanism, not by itself finish the original safety or deployment objective.

Independent selection/calibration/confirmation sources stay closed. Full-floor
harm is a separate diagnostic denominator, never a substitute for selected risk.
All results remain obs8/pred12, stride12 raw frames, image-local detector silver;
no metric, seconds, gold, physical-safety, true3D or foundation claim. Stage5C and
SMC remain disabled.
