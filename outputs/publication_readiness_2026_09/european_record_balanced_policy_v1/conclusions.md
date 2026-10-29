# Record-Balanced Constraint Experiment: Partial Fitting Repair

29 October 2026. **Complete; full-policy safety failed; no deployment promotion.**

## Question and Controlled Change

The preceding cost-sensitive policy learned useful interventions but violated
many recording-level constraints. I tested whether weighting each recording's
constraint equally, rather than by its utility weight, improves that failure.
The benefit objective, forecasts, causal features, architecture, initial states,
optimizer, 2,000 updates, dual update and 2% budgets were unchanged. Constraint
weights still sum to one. The change reallocates penalties; it does not relax
the scientific risk threshold or increase every penalty.

All 18 final neural policies were frozen in commit `7481dfd5` before readout.
Every registered seed and recording remains. There is no outcome-based seed,
checkpoint, threshold or recording selection.

## Results

|Fitting measure|Frozen cost-sensitive control|Record-balanced policy|
|---|---:|---:|
|Aggregate source/context/head views within budget|54/72|52/72|
|Required recording/context/head views within budget|84/444|131/444|
|Known all-positive-harm violations|346|289|
|Known easy-positive-harm violations|54|39|
|All-harm upper-bound violations|346|290|
|Easy-harm upper-bound violations|70|59|
|Undefined easy selected-risk views|14|9|
|Net easy-degradation violations|0|0|
|Eligible positive benefit retained|81.5516%|80.9251%|
|Selected row occurrences|293,781|292,961|

Of the recording views, 59 gain a budget pass and 12 lose one; 72 remain passing
and 301 remain failing. These are 444 repeated recording/context/head views,
not 444 independent recordings. There are 74 unique source-qualified recordings,
148 recording/forecast-context views and three policy-head seeds.

Required recording coverage improves for all three head seeds:
30 to 46, 27 to 40, and 27 to 45 passes out of 148 each. This consistency is a
fitting observation, not a confidence interval or unseen-scene result. The
forecast seed remains fixed; three policy-head seeds do not represent three
independently trained forecast systems.

## Utility Is Not Safety

Against the primary control, full-policy utility changes by
**[-0.135714, -0.133642] percentage points** of reference cost. At equal
intervention counts, the change is **[+0.020396, +0.021290] pp**. The latter is
a small utility-selection improvement, not a full-policy safety pass.

These intervals bound missing-label outcome completions on fitting data.
They are not bootstrap CIs, FDE improvements or independent test results.
The matched comparison with the older origin policy spans zero
([-1.124007, +1.104923] pp); superiority over that policy is not established.
No matched comparison is empty in this run. The code still treats empty matched
risk as `not_estimable`, never as a safe zero.

Net easy error stays within 2% everywhere, but that does not waive the separate
positive-harm/reference budgets. Average source coverage also cannot substitute
for required recording coverage. The approved prospective protocol is unchanged;
earlier failed experiments remain failed.

## Mechanism and Remaining Failure

372 recording/head views receive a larger constraint coefficient. Their budget
passes increase from 30 to 76 and known all-harm violations fall from 339 to 280.
The other 72 views receive a smaller coefficient: passes rise from 54 to 55,
but known all-harm violations increase from seven to nine. These post-fit slices
are consistent with a weighting effect, not proof of a unique failure cause.

285 of the 289 known all-harm violations remain in locality008/074. Immediately
before the last update, 286 all-risk constraints still have positive excess;
no all-risk dual is at its cap. Raising the cap alone is not a supported repair.
Convergence, shared-feature sufficiency and model expressiveness remain distinct
possibilities. This experiment does not isolate them.

The earlier exact TRAIN feasibility diagnostic also found a separate obstruction:
locality082/51 lacks an eligible easy-reference action within 2% under the fixed
candidate forecasts in two contexts. Reweighting a selector cannot invent that
missing action. The recording is retained, not excluded or relabeled safe.

## Execution and Provenance

- `fresh_run`: 18 real Torch policy fits, 36,000 updates, 72 fitting readouts,
  exact readout replay, arithmetic reconciliation and post-fit diagnosis.
- `cached_verified`: source-TRAIN packets, fixed forecasts, causal features,
  training-only moments, origin and cost-sensitive controls.
- `not_run`: new trajectory forecaster, new inner-held readout, independent
  selection/calibration/confirmation, deployment or full historical test suite.

Training PID28251 exited 0 in 732.451502s with peak RSS 2,821,308,416B. Readout
PID30534 exited 0 in 85.444484s; replay PID31124 exited 0 and reproduces all 72
views exactly. The reports also replay exactly and reconcile 1,836 policy
summaries. Targeted testing: 68 existing/current tests pass in 3.97s, followed
by three new post-fit-analysis tests in 1.37s. This is engineering evidence,
not an independent scientific validation.

Training computation was local native arm64 CPU4/interop1/workers0. Frozen TRAIN
packets were streamed read-only from CREATE and hash-checked in memory. No new
HPC training job was submitted. The marked CREATE queue snapshot at10:02:07UTC
listed zero current jobs for the account; it says nothing about historical job
completion. Data and checkpoints remain private.

## Next Research Step

The next priority is a TRAIN-only forecast/action-support repair, especially the
known unsupported easy-reference recording, rather than another threshold search.
Use a separately frozen comparison and retain every required recording. Do not
manufacture a pass with a zero/tiny pseudo-intervention or change the denominator.
For the remaining shared-policy fitting failures, retain the useful cost-sensitive
loss and investigate constraint convergence before adding a larger architecture.

That next experiment is **not_run**. Independent confirmation stays closed while
the fitting mechanism fails. Deployment remains unchanged; this result does not
promote the policy, validate historical Stage37 results on this new task, or
establish submission readiness. No Stage5C, SMC, metric/seconds, true3D or
foundation claim. The current task is eight observed and twelve requested future
steps in image coordinates, distinct from the historical raw-frame t+50 task.
