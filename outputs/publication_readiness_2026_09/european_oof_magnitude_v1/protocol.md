# Honest Fitting-Only Magnitude Readout

Registered before any new training or held readout. This is exposed source
development, not independent validation or risk calibration. Obs8/pred12 native
annotation steps and detector-image pixels; no metric/seconds, human-gold,
physical-safety, true3D or foundation claims. Stage5C and SMC stay off.

## Question and Controls

The prior-repaired auxiliary improves harm-presence ranking but not expected
harm severity. Test whether two bounded nonnegative magnitude slopes, learned
from honest inner out-of-fold (OOF) predictions, recover useful cost estimates.
Apply the identical readout to cost-only, repaired-true and repaired-shuffled
heads, and retain each identity control. Never compare only to a weaker raw
control. No architecture, threshold, trajectory predictor or checkpoint search.

Earlier frozen neural/fractional readouts and context residuals failed to show
consistent cost transport. This test is narrower: two origin slopes, matched
controls, new honest OOF repaired-auxiliary scores, not another context MLP.

## Complete Exclusion Lineage

For each of 144 outer views, three controller localities fit and the fourth
is excluded. Inner heads train on two of those localities and predict the
third. Auxiliary cap labels for these two-site heads require new single-site
cost references: rows on A use a reference trained only on B, and conversely.
The inner held locality C and outer held locality D are excluded from reference
fitting, preprocessing, easy-cut definition and labels. All controller
localities are disjoint from the frozen forecast-producer roster.

Do not reuse the older reference for A trained on B+C; that would leak C into
auxiliary training. Single-site preprocessing uses the exact same supported-row
sufficient statistics as its parent, in a new scoped helper. This is a nuisance
reference extension, not a change to the approved evaluation split.

144 unique single-site cost-only references, deduplicated only after exact
input/target identity checks. 864 fresh two-site auxiliary heads (true/shuffled).
432 matched two-site cost-only controls are cached_verified. All use width64,
2000 updates, three fixed seeds17/29/43, original site-balanced sampling,
CPU4/interop1/workers0, checkpoint200 and heartbeat200. No early stopping.

OOF target easy definitions use each two-site producer's fitting-only cut, not
the common outer cut. Report cut drift. The full three-site outer heads remain
frozen from the prior experiment; no refitting from exposed outer labels.
Single-site reference to two-site head to three-site head transport is a
limitation, not hidden independence or a calibration guarantee.

## Fixed Readout

For H_all and H_easy independently, fit origin least squares with equal
positive-envelope supported mass per fitting locality: a=sum(w*p*y)/sum(w*p*p),
clipped to [0,8]; use1 if a component has no predicted support. No intercept,
bins, feature stacking or selected tuning parameter. This minimizes the
unprojected objective only. Inference then projects H_all<=causal envelope and
H_easy<=H_all, preserving frozen denominator columns. Record fit and projection
behavior. OOF results are fitted diagnostics, not independent evidence.

## Readout and Gates

Freeze all outer predictions and commit before the new exposed-source readout.
Same primary: positive-envelope expected easy-harm MSE, not trajectory FDE/ADE.
Same guards: easy-harm top10 capture, coverage-log error, all-envelope H_all MSE.
Contrasts: scaledtrue/scaledcost, scaledtrue/rawtrue,
scaledtrue/scaledshuffled, scaledcost/rawcost, scaledshuffled/rawshuffled.
Advance only if full-input primary CIs are positive in all six assignments
against both scaledcost and rawtrue, true beats scaledshuffled in all six, and
no registered guard is negative or missing in the first two contrasts.
Three seeds average inside each of four held localities;3000 paired locality
bootstrap resamples per assignment. Overlapping assignments and exposed
source roles make all intervals descriptive and unadjusted.

## Operations

Use lossless gzip for this run's completed checkpoints, retaining full optimizer,
RNG and preprocessing state; partial checkpoints remain resumable. Keep at least
10GiB disk free, preserve unrelated staged changes, publish code and lightweight
reports only. Real pilot estimates time/space before the complete run. CREATE
was inspected read-only; no job submission. Independent selection, reserved
calibration and confirmation remain closed. No model promotion from this study.
