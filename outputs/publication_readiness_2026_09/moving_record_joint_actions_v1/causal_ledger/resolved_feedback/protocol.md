# Does resolved forecast feedback improve switching?

This is a new development experiment on the already exposed Moving A roster,
not independent confirmation. Fixed: 125,540 rows, 1,613 queries, 34 recordings,
14 sources, and forecast-bank seeds 17/29/43. Each bank is fitted outside A.
The task is eight observed to twelve future annotation steps in image coordinates.

At current frame t, match an agent to its forecast issued at exactly t-12
within the same recording. Score only that old endpoint against the current
observed position; never read the current or previous target arrays to build
inputs. Carry the latest resolved observation for that agent and a per-recording
decayed average of all resolutions (half-life 24 annotation steps). Zero with
an explicit support channel means missing feedback, not zero error.

Primary contrast: own+recording vs static, with frozen static risk estimates,
legal action set, and exactly the same switch counts at every query. Secondary:
own vs static, own+recording vs own, and free allocation with each head's
own risk forecasts. Fit the same actionwise ridge readout (0.002) excluding
the held source, with the same fitting moments and weights. The static head
uses the frozen original coefficients; both static arms must reproduce the
prior ledger action arrays exactly. Compute easy-risk-excess error on the
same original selected rows, including the existing missing-label charge.

Freeze features before reading targets, then freeze all actions before
scoring. Report every recording/seed, 3,000 paired source bootstrap draws,
completion-lower and known utility, and original all/easy 2% positive-harm
and signed-net/support rules. Undefined risk stays undefined. Empty matched
utility comparisons remain non-estimable. A gain here is development
evidence only. No new risk certificate, deployment, independent outcomes,
Stage5C, or SMC.
