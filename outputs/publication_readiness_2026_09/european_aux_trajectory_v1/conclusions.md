# Auxiliary Cost Deficits Are Already Visible Early in Training

I reconstructed the original experiment rather than changing another model
family. All 432 final numerical states match the original checkpoints.
The new evidence is the training path, not a new winning model or independent
replication. The auxiliary still does not earn a deployment promotion.

## Completed Experiment

Fresh native-Torch computation: 864,000 updates, 432 heads and 2,160 snapshots
at fixed updates 200/600/1000/1400/2000. Three seeds, six assignments, full and
motion-only features, and cost-only/true-event/shuffled-event arms are retained.
Registration `0dc819d7` preceded training; `6de58cf8` froze all results before
aggregate readout. Recorded fitting time is 807.07 seconds; run-phase wall
time is 1,518.36 seconds, excluding ancestry preflight and reusing the pilot.

Cached and hash-verified inputs include fitting features, nested targets,
normalization, row streams and original final states. No new held-out policy
readout, independent-role access, checkpoint selection or model deployment
was performed. The original first pilot's additional step-100 log is the only
allowed trace difference; prescribed numerical entries match exactly.

## Main Observations

For full inputs, true auxiliary four-cost fitting MSE is worse than cost-only
in all six assignments at every recorded time. Motion-only shows the same
point-estimate pattern. These are dependent comparisons, not 60 independent
experiments. Full-input cost-loss reductions are -2.79% to -1.60% at step200
and -2.63% to -0.80% at step2000. Negative reduction means increased loss.

| Update | Full cost4: positive / negative / overlapping intervals vs cost-only | Full positive-envelope easy harm: positive / negative / overlapping |
|---:|---|---|
| 200 | 0 / 5 / 1 | 0 / 1 / 5 |
| 600 | 0 / 6 / 0 | 0 / 2 / 4 |
| 1000 | 0 / 6 / 0 | 0 / 3 / 3 |
| 1400 | 0 / 6 / 0 | 0 / 2 / 4 |
| 2000 | 0 / 5 / 1 | 0 / 2 / 4 |

No full-input easy-harm interval is wholly positive against cost-only at any
recorded time. True labels do sometimes help relative to shuffled labels:
at steps200 and600, three of six easy-harm intervals favor true labels.
At step600, all six point estimates favor true over shuffled, while five
are worse than cost-only. Beating an irrelevant auxiliary is not evidence
that the auxiliary improves the strong no-auxiliary model.

For full-input true-auxiliary states, shared cost-gradient conflicts decline
from156/576 batches at step200 to120/576 at step2000. Easy-harm conflicts rise
from40/576 to63/576. At step200, assignment-level median auxiliary/main shared
norm ratios range0.874-1.942 for cost4 and1.443-4.599 for easy harm. These are
dependent batch diagnostics, not a proof of interference or a benefit from
norm clipping. The earlier negative projection experiment remains valid.

At step200, additive error accounting shows improved zero-harm fitting error
but worse positive-harm contributions in all six full-input assignments.
The late pattern is not uniform: P1/C2's final loss increase is dominated by
zero-harm rows, whereas P2/C1 has a substantial top-one-percent contribution.
Neither removing zero rows nor a generic tail-weighting claim is justified.

## A Separate Post-Hoc Hypothesis

Inspecting verified fitting metadata after the registered readout exposed an
initialization mismatch. The auxiliary starts at the old easy-membership
prior, median0.27994, but its cap-event target prior is median0.05091 for full
inputs and0.01081 for motion-only inputs. The weighted constant-prediction BCE
excess over the matching event-prior intercept has medians0.18409 and0.27455
nats, respectively. `initial_prior_posthoc.json` preserves all144 metadata
records and explicitly marks this check post hoc. It is not random-minibatch
BCE, a preregistered success test, or proof that changing the intercept helps.

The next discriminating repair should change this one initialization choice
under matched true/shuffled/control training, not select a convenient time
point or enlarge the architecture. No such repaired model was trained here.

## Limits

These are normalized fitting losses, not ADE/FDE gains. Three seeds and three
fitting contexts are averaged within each of four localities; 3,000 paired
locality resamples give descriptive, unadjusted intervals. Models and contexts
overlap. Of1,920 stratum/metric/contrast/time cells,420 are not_estimable under
the required complete locality/context support; none was silently discarded.
All primary cost4/all and easy-harm/positive-envelope cells are supported.

The first measurement is step200, so exact onset and causality remain unknown.
Late-only overfitting cannot by itself explain deficits already measured at
step200. This does not exclude later overfitting or a transport problem.
Independent selection, reserved calibration and confirmation remain unopened.
Obs8/pred12 native annotation steps, detector pixels; no metric/seconds,
physical-safety, human-gold, true3D or foundation claim. Stage5C/SMC stay off.
M3W is not yet submission-ready.
