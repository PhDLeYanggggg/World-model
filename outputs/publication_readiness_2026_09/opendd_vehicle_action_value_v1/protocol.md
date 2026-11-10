# Predict the value of an action, not a replacement trajectory

Keep the frozen two additive edits, each actor's displacement at every
step, and the original risk checks. A policy may keep both directions
or rotate one. Use the original twelve nonzero angles. The target
is the signed change in trajectory error against the exact unrotated
proposal. Split the observed step contributions into positive gain
and harm, charging each unknown step its maximum possible harm.
Predict the same lower change on incident relative trajectories.
Normalize by the actual action disagreement, not by future speed.

Use the same architecture, labels, optimizer and updates for ego
and context arms. Context adds observed neighbor history summaries
and the other selected actor's forecast displacement, not future
labels. Each donor's candidates use the existing forecast checkpoint
that excludes that donor and both downstream locations. Generate
three deterministic random legal anchor pairs per donor recording.
Apply to the actual fixed additive pairs on calibration and held
locations; sampled donor anchors need not have the incumbent policy's
distribution. Use 200 updates, batch 256, all three seeds.

Compare unchanged incumbent, ego score, context score with identical
utility rule, context with a relative-gain requirement and term,
and three controls fixed to the context score's rotating actor and
recording: ego angle, donor-average angle, and context joint angle.
Original calibration and held roles are unchanged. Only the calibration
location supplies a tie-complete confidence prefix that passes both
reference gains, original risk, and positive lower gain over the
incumbent. If there is no eligible prefix, do not rotate.

Freeze choices and thresholds before held scoring. Report both
uncalibrated contrasts and actual calibrated intervention counts.
Bootstrap seven locations, averaging directions and seeds before
3,000 draws. No new external outcomes, release promotion, change
to 2%, Stage5C, or SMC.
