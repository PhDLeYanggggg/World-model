# Hold recordings fixed and require agreement on the angle

This OpenDD development experiment reuses the source-excluded three-angle
utility head, forecast bank, recipients and two original intervention
masks from the previous experiment. It fits no new weights. Use both
registered quotas (526 and 1,368 rotations) and three existing seeds.
Retain the original first angle if the new rule does not replace it;
neither rule can change the recording or remove a planned rotation.

Against the first angle, compare the previously learned utility rule,
a trajectory-only rule, and an agreement rule. Decode the separate
trajectory benefit-minus-harm and relative-motion heads in their native
scales (candidate path displacement and the training pair envelope).
Compare each alternative with the first angle in those units. The
agreement rule chooses the largest minimum of the two predicted
increments, each divided by the query's largest candidate envelope.
An alternative is eligible only when both increments are positive and
its original-risk probability is no lower than the incumbent's.
If no alternative qualifies, keep the first angle.

Also test a calibrated version. In each direction, use the opposite
location to take the largest complete score prefix in which every
replacement passes the unchanged risk and dual-reference checks and
has positive direct conservative trajectory and relative-motion gain
over the first angle. Never use held outcomes to set a cutoff. Report
this version separately; fixed rotations that keep the first angle
are not new releases and the calibration is not a safety guarantee.

Count hindsight trajectory and interaction opportunities separately
and jointly, then jointly subject to the original risk and reference
checks. Use direct pair bounds against the first angle so missing
outcomes are debited once. Report the same three thousand paired
location-bootstrap draws as the preceding experiment. Report observed
endpoint gain, conservative trajectory and relative gains, harm counts,
strict passes and actual angle substitutions. The previous frozen
shortlist readout must reproduce. Do not open any new external outcomes,
change deployment, or execute Stage5C or SMC.
