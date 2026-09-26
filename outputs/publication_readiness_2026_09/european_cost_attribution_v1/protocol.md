# Frozen Cost-Factor Attribution

## Material Passport

Previous goal turn: progress. Parent 406fb9d6 completed 288 real Torch fits but
failed the strong-comparator gate. Its 55 public artifacts and 97 source
bindings are rehashed. This follow-up computes fresh diagnostics from frozen,
cached_verified models and arrays. No fitting, threshold search, policy
readout or independent-data access. It cannot reverse the parent gate.

## Fixed Diagnostic

Retain every source assignment, seed, full/motion-only pair and held locality.
Re-extract both conditional harm experts using only causal inputs and verify
their composition against saved scores. Let p predict easy membership E,
let m/n predict harm within E/not-E, and H be realized all-harm. Decompose:

```
easy_error = (p-E)*m + E*(m-H)
all_error = (p-E)*(m-n) + E*(m-H) + (1-E)*(n-H)
MSE = mean(first_term^2) + mean(second_term^2) + 2*mean(first_term*second_term)
```

The signed cross term is retained; components are not independent causal
effects. E and H are future-derived labels used only by this offline audit.
Label-assisted substitutions E*m and p*H are deliberately unattainable at
inference, not new models, formal upper bounds or deployment results.

Report both all-known and positive-disagreement populations. Unknown labels
are excluded; exact-zero CV error remains outside E under the inherited
definition. Include unweighted and severity-squared-weighted Brier diagnostics,
outside-easy error mass, outside-easy p<0.1 mass, and severity above its fitting-
only weighted 99th percentile. The 0.1 split is descriptive, not a candidate
deployment threshold. Full/motion-only populations differ.

Three seeds averaged within locality; 3,000 resamples of four localities for
each assignment. Repeated source roles/windows are dependent, historically
exposed and not multiplicity-adjusted. No favorable-role selection. Freeze
this diagnostic code before the current attribution readout. No model claim
is passed by a label-assisted comparison.

The next repair must be justified by the attribution. If oracle membership
still leaves high severity error, probability calibration alone is inadequate.
If membership-weighted error dominates, examine severity extrapolation and
cost-weighted reliability, not another threshold sweep. Neither observation
alone identifies a unique causal mechanism. Any stacked trainable head using
membership scores must use nested/OOF scores; end-to-end joint fitting needs
an explicit matched objective comparison.

Local native arm64 inference is sufficient; CPU4/interop1/workers0. Resume
completed groups by hash, preserve all prior weights, keep10GiB disk reserve.
CREATE queue observation is read-only; no new jobs, remote M3W path unverified.
Eight observed/twelve predicted annotation steps, detector-derived pixels;
no metric/seconds, human-gold, physical safety, true3D/foundation or submission
claim. Independent selection/calibration/confirmation stay closed. Stage5C
and SMC stay off. Routine audit is delegated; no further user audit required.
