# Matched vehicle interventions, with double-excluded training predictions

Use the fixed completion-trained vehicle predictor from the preceding
OpenDD development experiment for each held location and seed. Keep the
existing eight observed and twelve forecast samples, roster and seven
location folds. Do not read Songdo or any reserved confirmation data.

For cost training on a donor site, use a fresh completion forecast trained
on the other five sites, omitting both that donor and the outer held site.
The 21 unordered site pairs times three seeds give 63 forecast fits. Fit
the existing six-output gain/harm value model on those cross-fitted donor
predictions. Its targets are increments against CV and OLS4, positive
all/easy harm and the corresponding reference masses. Easy/hard cuts for
each donor target come from the same five sites that produced its forecast.
For held scoring, compute the cuts on its six donors. No policy threshold
or early stopping is selected on held sites. These are development folds,
not an independent calibration or confirmation protocol.

The query menu is CV, OLS4, median4 and the fixed neural path. For all
non-CV choices, shorten (never extend) to one quarter of the median
positive per-agent common peak. This is a fixed matched-budget diagnostic,
not a new safety rule. A query may make zero, one or two edits, leaving
at least one CV actor. At two edits it must have a common-family control.

- **Joint:** use the maximum positive minimum of summed two-reference
  incremental gains, subject to nonpositive predicted all/easy risk excess
  at the original 2% budget. Sum expected masses in native coordinates.
- **Matched independent:** rank each action by its minimum two-reference
  incremental gain. Fill the joint arm's exact count and radius slots with
  distinct actors, without the aggregate risk feasibility constraint.
- **Matched family:** same independent scores, count and slots, with one
  common action family per query. Use CV fallback when joint makes no edit.

Freeze all held predictions before calculating scores. Compare with CV
and OLS4, and score matched paired full-grid individual error and relative
displacement error on the same active recording views. Report the original
recording risk outcome on the full denominator and mark no-edit risk
undefined. Location bootstrap intervals use 3,000 draws after seed means.
An unestimated comparison with no edits is not a gain. The joint rule
tests risk constraint aggregation, not a learned pair or physical model.

There is no new split or change to the 2% risk threshold. Do not promote
this exploratory result or reopen external confirmation data to tune it.
