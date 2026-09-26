# Paper Evidence Note: Actionable-Support Risk Diagnosis

Status: source-development evidence; not independent confirmation, a new
theorem, a new forecasting improvement or a submission-ready main result.

## Candidate Results Paragraph

We diagnosed whether a mean-moment risk head learns harmful-event ordering or
merely separates forecast pairs that cannot incur excess error. Across six
ordered source assignments, three seeds and two frozen forecast pairs, we
trained 144 locality-excluded heads for 2,000 updates each. Preprocessing,
easy-event definitions and score bins were fitted only on the three training
localities in each fold. For the full pair, median harm-event AUROC was 0.792
over all supported rows but 0.486 over rows with positive causal forecast
disagreement. Nevertheless, the learned score improved top-decile harm-mass
capture over a disagreement-envelope control in four of six all-row paired
locality-bootstrap intervals. Harm magnitude remained unstable: 50 of 72
dependent locality views underestimated total harm, with median predicted-to-
observed mass 0.663. Thus aggregate event discrimination, costly-tail retrieval
and reliable harm-budget estimation are distinct properties in this setting.

## Why Both Ranking Results Can Be True

Let R and P be the frozen reference and candidate trajectories, and let e be
their mean waypoint disagreement. By the triangle inequality, the positive
excess ADE is bounded by e. Easy-event harm is therefore also zero when e=0.
The risk head enforces this support constraint with an envelope-multiplied
output. Such rows are structural negatives, not successful predictions of a
rare harmful event on an actionable pair.

Conditioning on the negative population partitions all-row AUROC into a
weighted comparison against structural negatives and a comparison against
negative rows with positive disagreement. The first can be nearly trivial.
Consequently, high all-row AUROC does not require strong conditional event
discrimination. This is a basic support/accounting observation, not a novel
statistical guarantee. Expected harm also weights severity; event AUROC and
harm-mass retrieval need not move together.

## Evidence to Retain

Use the complete results table, harm_ranking.svg and training_loss.svg.
The paired contrast averages three seeds within each held locality and then
uses 3,000 bootstrap resamples of four localities per source assignment.
Intervals are exploratory, unadjusted for multiple contrasts, and are not
independent confirmations. Keep the motion-only results and its 30/72
weak-support views beside the full-pair result. Do not present a favorable
seed or an all-row AUROC alone.

## Claims This Does Not Support

No controller was newly evaluated or deployed. Tail capture is not trajectory
accuracy, coverage of total harm is not a safety guarantee, and the all-row
bootstrap contrast does not certify low-score accepted actions. The new
three-site-cut inner populations and original whole-B-cut transport
populations do not define the same event. Their medians cannot establish a
causal improvement from crossfitting. Independent calibration/confirmation,
strong-method comparison and successful risk-matched policy evaluation remain
necessary for a main claim.

Current labels are detector-derived image-pixel coordinates and horizons are
annotation steps. No meters, seconds, human gold, physical safety, true3D or
foundation-model claim is warranted. Stage5C and SMC remain off.
