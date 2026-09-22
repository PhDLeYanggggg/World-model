# Equal-Capacity, Equal-Budget Cost-Objective Controls

## Material Passport

Source-only research development on four already explored SDD sites. The prior
width/duration study found positive protected gain for the width128/12000
intermediate-loss head, but its fraction comparator was width64/3000. This
comparison repairs that remaining capacity/budget mismatch. It cannot supply
independent confirmation or retroactively make the source sites unexposed.

## Frozen Comparison

Train24 new controls: native squared-cost and fraction squared-cost objectives,
four outer sites, seeds17/29/43. Width128,12000updates,batch256,AdamWlr.001,
weight decay.0001,clip5. Use the existing bounded_cost_head trainer unchanged.
The12 completed intermediate-loss wide12000 heads are cached-verified references,
not fresh fits. All heads have the same45,954 parameters and geometric output
bound. The forward expressions for bounded_native/bounded_fraction are identical;
only their objective weighting differs. No direct/unbounded head is introduced.

Retain the same causal356features, preprocessing, frozen EqMotion forecasts,
pair-excluded complete supervision, initialization seed and scene-uniform row
draws. Each final control's draw vector must equal its intermediate reference.
No incomplete target is sampled. New compute288000updates/73728000draws; cached
reference144000updates/36864000draws is reported separately. Resume pilot and
interrupted fits, preserve all parent artifacts, and never restart on a poll timeout.

All24 fits complete before aggregate readout. Net-stop, strict-stop and
matched-count rules remain unchanged; matched counts use the original frozen
fraction-strict anchor, not the new controls. Archive decisions before aggregate
outcome reduction. No label, future support or outcome-defined easy/hard flag
enters inference. No new normalization, goal construction, threshold selection,
calibration or original closed-role readout.

## Primary and Secondary Readout

The fixed candidate is intermediate/strict, not whichever arm wins. Its
objective-contribution check requires BOTH paired equal-site ADE contrasts
(intermediate minus native, intermediate minus fraction) to have lower3000-site-
bootstrap bounds above0. Also retain each seed's positive CV gain, aggregate
easy degradation<=2%, and complete exact-zero-CV harms0. Both contrasts are
reported even if one fails. This conjunctive development check is not a new
independent test or evidence that all subgroups are protected.

Report every arm/policy, ADE/FDE, all sites/seeds, hard/easy, tails, worst site,
selected/unknown/incomplete counts and full-grid incomplete-label gain bounds.
Retain the known deathCircleeasy failures of the intermediate reference.
Use training-defined q50/q90/q99 speed/disagreement strata for fitting and held
cost diagnostics. Report matched-count contrasts separately from fixed-threshold
gains. Secondary intervals are descriptive/unadjusted; no post-readout winner
promotion, loss mixture, exponent search or threshold repair.

Recompute all36 heads and replay1,581,804 score rows, verify cached intermediate
scores/choices/summaries exactly, then check labels, policies and reductions with
separate arithmetic. Report shared versus separately implemented checks honestly.
Three seeds and3000resamples of FOUR exposed sites do not establish independent
scene calibration or population safety. If the candidate loses to the fair
controls, preserve the loss contribution failure instead of using the old
undertrained controls as the paper's main comparison.

## Runtime and Claims

Nativearm64 CPU4/inter-op1/workers0, single-process lock, atomic500update
checkpoints and100update PID/loss heartbeat. Local cached-head fitting is
feasible; no new CREATE workload is needed absent changed resource evidence.
Eight observed/twelve predicted annotation steps, SDDstride12, annotation pixels.
No metric/seconds, true3D, foundation-model, deployment, calibrated-probability,
submission-ready, Stage5C or SMC claim. The larger research goal remains active.
