# Direct supervision for replacing a nominated angle

Keep the prior OpenDD development recordings, two fixed masks (526
and 1,368 rotations), recipients, clocks and top-three angles. For each
donor recording, use the source-excluded upstream forecasts, additive
chooser and original angle model from the shortlist experiment. The
donor and both outer sites are absent from those models' fitting sets.
Treat the first angle as incumbent and compare alternatives two and
three with it.

Fit two equal-capacity pair networks. They receive identical causal
incumbent and alternative features and displacement envelopes, use
the same donor rows, optimizer, batch 256, 200 updates and three seeds.
Only the regression targets differ. The anchor-difference arm
subtracts the two conservative gains against the unrotated anchor.
The direct arm computes one conservative bound on the replacement
itself, charging the possible movement on missing coordinates once.
Both predict two utility components and the same probability that
the alternative passes original risk and both reference checks.
Use normalized gains and no future information in inference features.

Choose the highest positive minimum of the two estimated improvements,
weighted by predicted pass probability. Keep the incumbent if either
gain is nonpositive for both alternatives. Report this choice and the
trajectory-only choice. As before, separately calibrate a score
threshold on the opposite location: the largest complete score prefix
where every replacement passes the original 2% risk and both reference
checks and has positive direct conservative gains on both measures.
Preserve original recording masks even when no replacements pass.

Freeze predictions and cutoffs before held scoring. Replay the original
margin, prior utility and prediction-agreement controls. Compare direct
and anchor-difference arms at both equal budgets and after calibration,
including direct paired lower bounds, relative motion, observed FDE,
swaps, harm, and 3,000 location-bootstrap draws. Compare training target
signs and report where direct and anchor-difference labels disagree.
OpenDD remains development; no new external outcomes, policy promotion,
Stage5C or SMC.
