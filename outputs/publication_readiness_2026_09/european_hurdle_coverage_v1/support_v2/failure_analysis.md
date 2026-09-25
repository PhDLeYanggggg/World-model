# Failure Analysis

Result source: fresh matched-count diagnostic over cached-verified forecasts,
heads and source identities. No new model fitting, threshold search or deployment.

| Explanation | Evidence | What Is Not Established |
|---|---|---|
| Hurdle supervision universally improves risk ordering | Rejected by the matched-count matrix. Neural/all has 3 positive and 4 negative all-ADE intervals at product counts; damping/all has 0 positive and 8 negative. | No universal ranking or neural dynamics benefit. |
| Every improvement is just switching less | Rejected. All-event coverage components are positive in all nine views for both candidates along both paths. | Counts are not a sufficient policy: which rows are changed still matters. |
| The easy-event protection is entirely better ordering | Unsupported. Hurdle at product counts exceeds the positive-easy limit in 6/9 neural views; product at hurdle counts passes it in 9/9. | The decomposition is exact accounting, not a unique causal mediation estimate. |
| Fixed predicted risk guarantees zero-reference protection | Rejected in the parent controls. Neural hurdle still harms zero-CV cases in 12/18 full views. | Passing the positive-easy percentage limit does not protect zero-error cases. |
| Unequal support can be ignored | Rejected by the pre-readout guard. Two groups require separate full/common accounting. | Tiny observed aggregate support effects do not justify silently filtering them. |
| One good fold establishes stable ranking | Unsupported. Neural/easy positive point contrasts occur in fold2; other folds reverse direction. | These overlapping development views are not independent replication. |
| Matched-count arms can be deployed directly | False. All nine high-count neural/easy arms violate their predicted-risk rule. | Their accuracy/safety readouts are offline diagnostics only. |

The earlier support audit found four distinct zero-CV rows, all in one locality;
folds evaluating them had none in fitting. This parent evidence is hash-verified,
not a new support discovery here. A model cannot demonstrate transfer on an
unrepresented event merely because the average easy metric is acceptable.
Memorizing that readout locality or those rows would not be a legitimate repair.

## Repair Direction

The next controlled method change should target ordering separately from score
scale: fit a ranking-aware gain/harm objective on fitting-only cross-fitted
predictions, retaining the product/hurdle controls and both count diagnostics.
Support-aware abstention should use causal history signatures with training-only
support estimates, not future CV error or hand-picked readout IDs. Separate this
from risk calibration so a coverage reduction is not mislabeled a ranking gain.

A repair must show improvement over the equally protected damping comparator,
including easy, zero-reference, tail and worst-locality errors. Changing the
denominator, risk tolerance, bootstrap unit or selected fold after outcomes is
not a repair. Unknown future labels remain unknown, never assumed harmless.

## Limitations

The forecasts were not retrained; this is a controller diagnostic. All twelve
European Squares localities have been opened for development. Each excluded
eight-locality readout is excluded from its own fitting chain, not independent
final confirmation. The three seeds share data; overlapping windows are not
independent samples. The 3,000 paired locality resamples condition on fitted
models and do not correct repeated comparisons. No seconds, meters, human-gold,
physical-safety, true-3D, foundation or Stage37 recertification claim is supported.
No reserved data, Stage5C execution or SMC was used.
