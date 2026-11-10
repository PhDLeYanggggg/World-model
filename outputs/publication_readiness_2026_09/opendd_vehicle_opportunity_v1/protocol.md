# Select the recording before changing a direction

Keep the OpenDD vehicle roster, eight observations, twelve predictions,
two additive recipients and all per-step displacements. Use the
previous ego action head to nominate exactly one legal rotation
per recording, including when it recommends abstaining. The
candidate cannot change between screening methods.

Train a recording-level head on whether at least one legal
rotation meets the earlier strict finite-menu criterion: both
references and original risk pass, no conservative trajectory
loss versus the original, and positive conservative relative
gain. Three donor anchor pairs per recording use the frozen
forecast excluding the donor and both downstream locations.
No nominee outcome, held label, future support or scene identity
enters the model features. Use three seeds and 200 fixed updates.

Freeze and compare ego action confidence, existence probability
and their fixed product (probability times sigmoid of ten times
action margin). Report both opportunity AUROC/AP and nominee
AUROC/AP. The diagnostic chooses the same number of recordings
for each score, equal to the previous ego policy's recording
count in that direction, and also a common 15% quota. Quotas are
offline comparisons, not releases. For deployable simulation,
fit a tie-complete prefix using the separate calibration location
and the original risk, dual-reference, and positive incremental
gain checks; report actual held selections, not fallback passes.

Average within location across directions and seeds, then use
3,000 paired bootstrap draws over the seven locations. This is
development evidence; no new external outcome, release, risk
limit, Stage5C or SMC.
