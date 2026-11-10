# Predict whether the nominated change is worth releasing

Keep the existing eight-observation, twelve-prediction OpenDD vehicle
roster, queries, two additive edits, and displacement norms. Held
and calibration nominees are the identical ego actions from the
recording-screen comparison. A ranking model cannot change the
angle, actor or number of original edits.

Generate training nominees with an ego head excluding the target
donor and both outer locations. Its training examples use a
forecaster excluding those three locations and each respective
inner training source. The held and calibration labels are never
used for either head fit. The actual nominee is predicted before
its donor label is computed. Use the previous three random legal
anchor pairs per donor recording, not held incumbent pairs.

Fit two identically sized heads to the exact nominee's strict
success: original risk and both references pass, conservative
trajectory utility is preserved, and relative motion improves.
Use the same 200 updates and the three registered seeds. One head
uses binary cross-entropy. The other adds a within-source loss
against the highest-scoring 15% of negative nominees. Both see
the proposed action, its margin, and the past-only menu summary.

Compare the two heads with the frozen action margin and earlier
opportunity score. Freeze all proposals and calibrate on the
opposite excluded location using the original risk, dual-reference
and positive incremental utility criteria. Report matched previous
ego count, matched 15% quota, and separately calibrated rotations.
The quota is an offline diagnostic. Use 3,000 paired location
draws after averaging directions and seeds. OpenDD is development;
no external outcome access, new deployment, changed 2% limit,
Stage5C, or SMC.
