# Safe Choices Exist, but the Current Policy Does Not Recover Them

## Material Passport

Code-experiment diagnostic, completed29October2026. Fresh_run: exact constrained
oracle and frozen-policy replay on fitting recordings. Cached_verified: existing
data, forecasts, moment predictions and intervention heads. Not_run: new held
evaluation, independent confirmation or deployment. This is not a model success.

## Findings

The verified cohort contains299,020 fitting row occurrences,298,325 causally
eligible rows,24 source/context packets and74 distinct source-qualified recordings.
Each recording appears in two forecast contexts, giving148 record/context views.
Three head seeds produce72 control views and444 repeated record/head views;
these repeats are not independent data.

An exact label-aware selection with positive gain and supported2% all/easy
budgets exists in **146/148** record/context views. The two exceptions are the
same recording, locality082/51, under its two frozen forecast contexts. It has
74eligible rows and only3known eligible easy-reference rows. None of those three
can anchor a2%-bounded selected-easy risk in either context. Mixing exclusively
over-budget easy ratios cannot resolve this. That is a fixed-action/support
obstruction, not a failure that further selector optimization alone can remove.

The eligible known positive-benefit upper bound averages19.662percentage points
of each packet's known reference cost. This is a label-aware fitting bound, not
an achieved forecast improvement or a complete-roster safe policy.

|Frozen policy|Actual fit budget views|Actual record budget views|Safe positive deletion-oracle views|Eligible positive benefit retained|
|---|---:|---:|---:|---:|
|Origin|1/72|16/444|337/444|43.80%|
|Exact whole-record gradient|53/72|178/444|388/444|26.00%|
|Same-record meta-update|51/72|170/444|389/444|28.01%|
|Cross-record meta-update|51/72|160/444|371/444|23.46%|

The cross-record policy still contains a useful safe subset in371record/head views,
but its actual selected set meets the budget in160. It also discards76.54% of
eligible positive benefit. Thus training-choice quality remains unresolved in
addition to the one-record action/support obstruction. These facts do not show
that the oracle choices are predictable from causal features.

## Execution and Verification

Corrected real diagnostic PID18982 exited0:85.096418s, peak RSS2,970,566,656B.
Replay PID19475 exited0 and exactly reproduces all72 views and summary. Every
frozen complete-view and per-record policy score matches its previous verified
readout. No optimizer step occurred. Seventeen current targeted tests pass2.09s,
including the oracle tests and new cost-sensitive prototype tests; the oracle
subset alone has11tests and240random exhaustive small-case checks. The full
historical suite was not rerun.

Version1's new grouped-control diagnostic had a numeric/string recording-ID bug.
It is preserved and marked invalid, not silently overwritten. Version2's regression
test reproduces the original failure and then passes; counts and prior per-record
scores are now independently checked. Earlier model results were not modified.

CREATE was queried read-only at08:58:43Z. The command returned0 with empty stdout
and an authentication advisory, without a remote execution marker. This does not
establish a fresh queue inventory or completion of historical jobs. No remote
job was submitted, cancelled or restarted. Local data and memory were adequate.

## Next Controlled Repair

Change only the neural primal surrogate to cost-sensitive logistic training,
keeping fixed forecasts, causal inputs, hard recording constraints,2% budget,
initialization, final threshold and2,000update budget. Compare18new policy fits
with the frozen exact-gradient control. Do not use a label-aware oracle at
inference and do not choose thresholds on held data. First answer whether the
fitting failure is reduced; no new held-data evaluation is included.

The irreducible locality082/51 obstruction stays explicit. A later forecast/action
or labeled-support repair is required there; fitting success elsewhere cannot
be called complete policy safety. Independent calibration/generalization and
scene-joint consistency remain separate paper gaps. No Stage5C/SMC or unsupported
metric,seconds,true3D,foundation or submission-readiness claim.
