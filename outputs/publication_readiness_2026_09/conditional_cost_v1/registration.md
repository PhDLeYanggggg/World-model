# Fixed Decision-Region Cost Weighting

## Material Passport

Prospective single-factor fit repair motivated by the completed matched-budget
diagnosis. The four SDD source sites remain research-design exposed. Original
val/test/main/external/bookstore roles stay closed. This is not independent
calibration, a final test, or a novel-risk-guarantee claim.

The frozen three-objective diagnostic finds near-neutral or conservative mean
harm on the full fitting-eligible population, but underprediction on the head's
own selections. Intermediate underpredicts8/12 fitting views and12/12 held views;
native underpredicts12/12 even in fitting. Held conditional shift remains larger.
This motivates testing finite-capacity fitting allocation, not global scaling,
another failure-event head, or a threshold sweep.

## Intervention

For each existing source/seed view, use its frozen intermediate128/12000head to
score only the cost-fitting inputs. Freeze the exact existing strict-stop mask:
positive forecast disagreement, nonzero last past displacement, benefit>harm,
harm<=.1benefit. Assign loss weight4 to that causal mask and1 elsewhere, then
divide all weights by their expectation under the existing scene-uniform complete
training distribution. Unknown-label rows have zero training weight and are never
sampled. The same multiplier applies to both benefit and harm squared errors.

Train12 new heads from the same seeded initialization with intermediate error
weighting(error squared/D), width128,12000updates,batch256,lr.001,weightdecay.0001,
gradientclip5,45,954parameters. Keep356causalfeatures, targets, preprocessing,
upstream pair-excluded forecasts, row draws and inference policies unchanged.
No new feature, model capacity, future-defined inference filter or loss exponent.
The multiplier4 is one fixed moderate emphasis, not an optimized hyperparameter.
No sweep follows a failed result. State-dependent weighting does not change the
unrestricted conditional-mean target; only finite fitting allocation is tested.

Reference-generated fitting weights are in-sample supervised development tools,
not out-of-fold calibration probabilities. Their predictor excludes the outer
scene. No outer labels or future support determine selection masks. Complete
labels are used only for fitting/supervision support and diagnostic reductions.

New compute144000updates/36864000draws; retain36old objective heads as
cached_verified references, not fresh fits. One100-update pilot is resumed.
CPU4/inter-op1/workers0; atomic500update checkpoint/PID-loss heartbeat100updates.
No new CREATE job is needed for this locally feasible head experiment.

## Fixed Evaluation

Complete all12fits before aggregate readout. Freeze scores/decisions first.
Use unchanged net-stop,strict-stop and matched-count policies. Matched count
retains the original frozen fraction strict anchor, not a newly picked rate.
Primary is new strict versus the completed intermediate strict reference, not
the weaker fraction arm. Positive paired3000site-bootstrap lower contrast bound,
positiveCVgain in everyseed, aggregate easy<=2% in everyseed, **each scene/seed
easy<=2%**, and complete exact-zeroCV harms0 are all required for the combined
development gate. This explicit per-scene condition cannot be hidden by an
aggregate pass. The current reference fails it; that does not lower the standard.

Retain comparisons with native/fraction, all policies, scenes/seeds, ADE/FDE,
hard/easy, exact-zero harms, tails/worst site, switching counts, missing futures
and full-grid gain bounds. Report same-population cost means on original and
newly selected regions in fitting and held data. A reduced training residual
without held gain/protection is a failed repair. Secondary CIs are descriptive
and unadjusted. Replaying the same model is not a new independent experiment.

Full new-checkpoint score replay, training-weight/sample validation and separate
label/policy/error arithmetic follow readout. Bootstrap units are four physical
sites, not overlapping windows or3000independent observations. Eight observed/
twelve predicted annotation steps, SDDstride12,annotationpixels; rawt50separate.
No metric/seconds,true3D,foundation,deployment,Stage5C orSMCclaim. Independent
scene risk calibration and confirmation remain required beyond this experiment.
