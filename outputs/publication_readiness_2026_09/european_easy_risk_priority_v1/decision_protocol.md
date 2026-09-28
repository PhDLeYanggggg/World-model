# Causal Action and Readout Implementation

This implements, without changing, the comparison registered before training
in f22e98c8. Its separate code hash is frozen before any new action or held
readout. Current real pilot may be queued/running; registration is not a result.

Restore only216 owned, hash-checked CREATE checkpoints after full training and
first-pair replay complete0:0. Retain10GiB local reserve plus800MB projected
checkpoint/action allowance; do not delete prior artifacts. Uncapped states must
match old supervised states. The two new arms share initialization and samples.

Future fields in the existing source archive are removed at the prediction
interface. All contexts and predictions receive only sites,history,origin,
geometry,recordings,frames. The new neural inputs remain causal context,
baseline descriptors and the causal forecast-disagreement envelope. The archive
loader may load targets into memory; they are not passed to action construction
or consulted for selection. No claim that archive bytes are physically unread
is made. Test with poison target objects and reject extra action-context keys.

Existing raw policy, old supervised scores and uncapped own-count actions must
reproduce exactly. The new common count may differ because the feasible-anchor
intersection changes; do not compare historical common-count policies directly.
Freeze all108 new action groups and commit their manifest before evaluation.

The registered primary is risk_priority_matched versus uncapped_matched ADE.
Report its nominal3000-resample locality interval, full-floor harm diagnostic,
selected-risk ratio including undefined abstention, easy/zero-CV harm, quality,
tails and all controls. The exploratory screen retains the preceding matched
count/ADE/full-floor-harm checks plus every-view selected-risk/easy/zero-CV
checks; a full-floor diagnostic cannot replace the original selected-risk primary.
No automatic deployment or independent-confirmation claim even if these pass.

Replay all actions and numerical readout before sealing results. Synthetic
loader/gate tests are not a real readout or full end-to-end verification.
Independent selection/calibration/confirmation stay closed. Image-local
detector silver, raw-frame obs8/pred12; no metric/seconds/physical-safety/true3D/
foundation/submission claim. Stage5C and SMC disabled.
