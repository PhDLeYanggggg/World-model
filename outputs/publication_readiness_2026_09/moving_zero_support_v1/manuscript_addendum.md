# Manuscript Addendum: Conditional Support and Annotation-Time Limits

This is a source-bound addition to the working evidence manuscript, not a new
submission-ready paper or a replacement for frozen earlier evidence packages.
All findings below come from development-exposed SDD sites.

## Rare Reference-Exact Events

An explicit conditional zero-reference component did not repair the protected
forecast policy. To distinguish numerical artifacts from insufficient support,
we traced all seven moving zero-CV cases to released annotations and compared
their past-conditioned representations with the actual source fitting rows.
These cases span three recording-scoped tracks in two recordings. Native
last-step motion is 6--16.5 annotation pixels; the complete eight-step history
is not exactly constant-velocity. Cached zero-error labels agree exactly with
raw coordinates on the specified twelve-step prediction grid.

Across 126 case/feature/action/seed comparisons, no moving zero-event training
row occurs among the nearest 512 source neighbors, whether using standardized
history coordinates or the complete risk representation. Relevant source
support consists of only 2--7 windows from 1--3 tracks per fitting view.
The cases nevertheless have nearby ordinary moving contexts, often with
beneficial neural forecasts. Thus density of general motion examples is not
equivalent to coverage of the event that a safety decision must recognize.

No exact source-feature collision was found for the seven cases. The analysis
therefore does not establish intrinsic prediction impossibility or a Bayes-error
lower bound. It exposes a limitation of the current finite support and empirical
leaf-frequency readout, without identifying data scarcity as the sole cause.
Neither a low population Brier score nor a feasible predicted-risk constraint
should be presented as rare-event protection or independent calibration.

## Supplied Annotations Versus Online Observations

Zero CV error is exact on the approved sampled grid, but not on the intervening
raw annotation frames, where ADE is 0.5--0.8125 pixels. All future sampled points
in these cases are flagged generated. Six histories contain generated points
bracketed by a later annotation control beyond the query time. This provenance
observation does not prove the annotation software's actual execution path, but
prevents a claim of verified online sensor-as-of input causality.

We distinguish a model that uses only supplied past annotations from a model
whose observations are known to have been produced without future information.
The former is the scope established here. Future control metadata and dense-grid
errors are retrospective audit quantities only; they are never supplied to the
forecast policy. We do not change the main observation/prediction grid, censor
the seven cases, or relax the risk requirement after inspecting this diagnosis.

## Consequences for the Main Claim

The current work can report reproducible baseline-relative source tradeoffs and
specific failure mechanisms. It cannot claim an independently calibrated
physical safety guarantee, a deployable new zero-event head or cross-dataset
foundation-model capability. Independent physical scenes with auditable source
processing remain necessary before evaluating the intended calibrated method.
More model variants on the same exposed source sites would not close that gap.

Reproduction, complete numerical tables, source hashes and the separate SciPy
arithmetic check accompany the [experiment report](conclusions.md). No new
predictor is trained and no independent confirmation outcome is opened.
