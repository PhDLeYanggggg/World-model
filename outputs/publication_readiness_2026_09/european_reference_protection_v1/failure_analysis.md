# Failure Taxonomy

## 1. Freezing Worked; the Repair Did Not
All protected reference outputs remain bit-identical to the original mean
head. Independent replay checks the stored frozen weights, matched initial
weights, every continuation draw count and final sampler state. The main
negative result cannot be attributed to accidentally training the reference
branch or giving one arm a different optimization budget.

Full protected-joint versus continued-joint has six all-ADE intervals spanning
zero. Complete observed risk passes are 5/18 versus 4/18, not a reliable repair;
the original mean joint passed 6/18. Motion-only protection is worse for this
criterion: 6/18 versus 9/18 continued and 12/18 original.

## 2. Training Fit Improves but Selected Harm Does Not
The full B population, not only a small training diagnostic batch, is evaluated.
Protected easy-harm MSE improves versus the original head in 18/18 full fits,
with median ratio 0.94647. Relative to matched continuation it improves in
17/18, median ratio 0.98957. Thus there is a small fitting benefit to isolating
harm learning. It is not sufficient evidence of a useful deployment policy.

With the same old raw-neural intervention mask held fixed, full-B predicted
easy-harm mass divided by actual mass falls from median 0.61847 originally
to 0.55643 continued and 0.54905 protected. Lower population squared error
therefore coexists with worse aggregate underprediction on the relevant
selected population, even on the fitting source. The problem is not solely
unseen-domain shift or a policy choosing different agents.

## 3. The Fitting Benefit Does Not Transport
On C, protected full-population easy-harm MSE improves over the original in
only 4/18 settings (median ratio 1.00842) and over continuation in only 3/18
(1.00226). Full fixed-action C harm coverage falls from 0.58180 originally to
0.53984 continued and 0.52150 protected. Motion-only protected easy-harm MSE
likewise improves over the original in only 5/18 C settings.

These are consistent with a mismatch between population MSE, selected-group
cost and cross-locality transport. They do not prove a single cause such as
gradient interference, lack of capacity or irreducible label noise. The prior
sampling study also failed, so repeating generic oversampling is not an
untried remedy. More continuation alone has now been tested as well.

## 4. Point Harm Estimates Remain Optimistic
On protected-joint's own C action sets, supported easy-harm predictions are
below actual harm in 66/72 full dependent locality views. Median predicted/
actual mass is 0.37512. This is a different action set from the fixed raw mask,
so its ratio cannot be presented as a paired calibration improvement.

The worst full example is locality074, fold1/seed29/controller2:

| Quantity | Value |
|---|---:|
| Actual selected easy harm | 2,020.01931 |
| Predicted supported selected easy harm | 203.90564 |
| Actual easy reference mass | 32,694.64004 |
| Predicted whole-query easy reference mass | 38,994.14243 |
| Actual positive-harm/reference ratio | 6.17844% |
| Predicted whole-query ratio | 0.53184% |
| Unknown-row fraction of predicted reference mass | 1.37666% |

The guard preserved an already imperfect reference estimate. Numerator
underprediction remains substantial; unknown future support alone is not a
plausible explanation for this magnitude. Future support and these observed
errors are diagnostic labels, never added to the inference rule.

## 5. Accuracy and Risk Must Not Be Conflated
Protected joint versus independent dual gates has positive all/hard intervals
across six assignments, but risk passes drop 14/18 to 5/18. The hash comparison
matches per-query intervention counts, not realized harm. It cannot establish
a better risk-accuracy frontier. Net easy passes all full settings while
positive harm fails most; net cancellation is not individual harm control.

The secondary protected-dual accuracy effect is retained, not promoted over
the failed primary joint-policy endpoint. No best seed, checkpoint or source
assignment is selected after readout.

## 6. Remaining Evidence Limits
Four localities per C role, three seeds, repeated source assignments and
historically opened development data limit inference. No independent
calibration, confirmation, new forecaster, physical interaction guarantee or
public strong-method superiority is established. The two-network branch adds
cost without resolving the primary problem. No deployment changes.

All values are detector-derived image-pixel annotation-step results. No
metric/seconds, human-gold, true3D, foundation or submission-ready claim.
Stage5C and SMC remain off.
