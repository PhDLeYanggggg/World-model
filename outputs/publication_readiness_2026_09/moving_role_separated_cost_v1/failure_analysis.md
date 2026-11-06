# Role-Separated Costs: Better Utility, Inadequate Selected Risk

## What Changed And What Did Not

This is a fresh matched model experiment, not a changed evaluation rule. I fit
48 Torch heads and repeated all 48 complete fits exactly. The two arms share
architecture, causal features, source-excluded fitting pools, normalization,
initial weights, sampling draws and 4,096-update budgets. The treatment separates
observed conditional-cost learning from label-support learning; the control fits
their composed moments. Both use the original six nonzero forecast amplitudes.
All 18 context/arm/seed decisions were frozen before the policy readout.

The original per-recording all/easy positive-harm, signed easy-net and reference
support requirements remain. Empty controls are NA. There is no identity-fallback
certificate, new threshold, removed view or new held C/H evaluation.

## Positive Registered Evidence

The registered joint-free contrast, role-separated minus composed-anchor, is
+0.445303 percentage points in paired completion-lower gain, with exploratory
source-block 95% interval [+0.090835, +0.930796]. The corresponding known-label
contrast is +0.527832 pp [+0.193976, +0.996965]. These are matched trajectory
utility results, not risk certificates. Separate completion lower gains cannot
be subtracted to obtain the paired completion contrast.

All three risk-head seeds have positive lower contrasts: +0.410917, +0.533530
and +0.391464 pp. Six of eight source means have positive lower contrasts.
Source065 is negative on that lower contrast; source073 is negative even on
known labels. The eight source components, not overlapping windows or 72
repeated recording views, are the bootstrap units. The forecast teacher has
only one fixed seed, and all own-B sources were exposed development data.

## Why This Is Still A Failed Safety Candidate

The role-separated joint-free policy improves known ADE by 0.667496% versus
causal CV, but passes only 23/72 required recording views. It has 27 known-easy
positive-harm violations and seven undefined easy-reference views. This is not
an acceptable deployment result, even though the mean trajectory contrast is
positive. No favorable source or seed substitutes for the full roster.
The same policy also has three known-all positive-harm violations,12 signed
easy-net violations,10 completion-all violations and42 completion-easy
violations. These overlapping counts are retained, not replaced by average gain.

The mutually exclusive failure categories are six missing-support views,
27 observed-harm violations, 16 completion-only violations and 23 passes.
Observed-harm failures take priority, so the six support-only failures need
not equal the seven views with an undefined easy ratio.

## Mechanism Diagnostics

1. **Target interference is partly repaired.** Within the matched five-moment
   family, observed conditional MSE falls 1.348588 to 1.189763; observed composed
   MSE falls 1.415037 to 1.308119. Bound composed MSE worsens 7.063742 to 7.661082.
   These share a known-only fitting RMS, not the old signed-risk scale. Support
   AUROC improves 0.539627 to 0.679486 and ECE 0.058111 to 0.038312. The label is
   released-label availability, not physical visibility or baseline failure.

2. **Full candidate coverage helps, but is not enough.** The separated proposal
   rule has 43 undefined easy views; full-bank independent choice reduces this
   to seven and improves paired completion-lower utility by 0.162274 pp. A veto
   of the frozen accepted mean/unanimity set cannot create missing reference
   mass, but a new veto over the broader proposer could. Full-action expansion
   is an empirical choice here, not a proved necessary construction.

3. **Joint optimization amplifies selected-cost optimism.** Separated independent
   choice passes 28/72 views, with 13 observed-harm failures. Joint-free utility
   improves, but support drops to 23/72 and observed failures rise to 27. The
   joint policy overpredicts selected easy reference in 61/72 views and
   underpredicts selected easy harm in 35/72. Action-invariant reference inputs
   prevent action-dependent denominator gaming, not a wrong reference value.
   Predicted query-level budgets are feasible; that does not certify realized
   per-recording ratios after selection.

4. **Missing labels are not the sole cause.** Seven separated joint easy
   violations select no unknown rows. Better support classification alone
   cannot repair these observed errors. Mean fit quality does not establish
   lower-tail reference accuracy or upper-tail harm accuracy in selected sets.

5. **The old conditional control remains informative.** On exactly the old
   known-proposal cohort with its frozen fitting RMS, conditional signed MSE is
   1.609501 for the old head, 13.721483 for the new anchor conditional branch
   and 2.810864 for the separated conditional branch. Thus separation improves
   this new family but does not recover the old conditional fit. Comparing the
   old conditional output directly to a new composed output would mix targets;
   conditional_risk_probe.json explicitly avoids that mistake. Architecture,
   coherent decoder, full-action fitting coverage and five-target loss differ
   from the old family, so this is not an isolated loss effect versus that head.

6. **The joint control is not pure agent membership.** Matched-count decisions
   preserve each nonempty query's intervention count, but can change both agent
   membership and amplitude. Empty utility controls remain not estimable. Neither
   the matched utility gain nor solver feasibility is a safety-gate replacement.

These are diagnostics of frozen choices, not a new policy search or a proof
that any one factor alone causes the failures. Per-view moments, amplitudes,
source and seed contrasts remain in the linked JSON/CSV artifacts.

## Next Model Repair

The highest-value next contrast is selection-aware, source-held reference/harm
uncertainty modeling, with a fixed proposal control and the full-action control.
Use fitting/calibration roles that exclude the query source; account for policy
selection rather than fit a single pooled ratio. First inspect the earlier
reference-subsidy/calibration negatives so this is not another renamed repeat.
Keep the original contract and register the comparison before its readout.
Do not merely tighten a global threshold, train a larger support classifier,
choose the best development seed or open a fresh held test to rescue this result.

## Claim Boundary

fresh_run: preparation, 48 fits, 48 full exact refits, complete causal allocation,
evaluation, paired source bootstrap and selected-risk diagnostics.
cached_verified: parent forecasts/proposals, donor features, source roles and old
controls. not_run: independent C/H confirmation, three forecaster seeds, new
trajectory dynamics, deployment. This is MovingDroneCrowd++ native obs8/pred12
image-coordinate available-label ADE, not legacy SDD t50, official complete-12
ADE, meters or seconds. Original24-source-group main remains NA, not passed.
Annotation-generation causality is unverified. No true3D, foundation, human-gold,
submission-ready, Stage5C or SMC claim.
