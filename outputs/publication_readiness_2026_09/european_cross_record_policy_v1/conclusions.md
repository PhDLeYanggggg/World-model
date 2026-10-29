# Cross-Recording Meta-Training: Complete, Not Deployable

## Result

The fixed cross-record objective did not repair policy safety or establish a
utility advantage over its matched same-record control. Keep deployment unchanged.
This is a completed negative experiment, not a runtime blocker or a completed
paper goal. All previous failed results remain unchanged.

36 real neural policy fits completed 72,000 updates on native arm64 CPU in
3508.267798s (58.47 minutes), peak RSS 3,687,137,280B. No crash, reduced scope,
NumPy fallback, held-checkpoint selection or threshold tuning occurred. These
are intervention heads, not newly trained trajectory forecasters.

## Main Comparisons

|Policy|Fit budget views|Inner-held budget views|Inner-held recording budget views|
|---|---:|---:|---:|
|Frozen exact gradient|53/72|0/72|2/186|
|Same-record meta-update|51/72|0/72|2/186|
|Cross-record meta-update, primary|51/72|0/72|4/186|

The primary has 63 known all-risk violations, 59 known easy-risk violations,
50 easy-degradation upper-bound violations, and 3 undefined easy ratios on 72 held
views. Every source slice is 0/6 budget views; every head seed is 0/24. Passing
4/186 individual recording views cannot rescue the complete-policy requirement.
The frozen origin also fails this cohort's budget; it is not newly certified safe.

Held count-matched utility completion bounds, in percentage points of reference cost:

- Cross minus same-record: [-0.188814,-0.044687].
- Cross minus frozen exact-gradient: [-0.159391,-0.031468].
- Cross minus frozen origin: [-1.571132,-0.745989].

These are missing-outcome completion bounds, not CIs or FDE gains. Full-policy
cross-minus-same bounds [-1.062365,+0.162885] include zero. Seven cross/origin
matched controls are empty; their risk remains not_estimable, not safe zero.

The fixed 3,000-draw source-cluster bootstrap adds uncertainty across 12 sources.
Its matched outer envelope is [-0.453565,+0.063353]pp versus same-record and
[-0.333954,+0.036209]pp versus exact-gradient. These include zero, so the negative
point comparison does not establish a universal meta-training disadvantage.
Versus origin, the envelope remains negative [-2.342835,-0.232816]pp. All are
exploratory, conditional on the fixed forecasters, not independent confirmation.

## Why This Matters

The cross-record policy retains 47.151% of origin's known benefit and 46.610% of
its known harm on held recordings. Same-record retains 59.213% and 57.595%.
The new objective mainly changes how many interventions survive; it does not
establish the needed separation between useful and harmful choices. Counterparts
have matched initialization, episodes, row exposure and update count. More
compute than the older exact-gradient control is disclosed, not presented as
an efficiency gain.

## Evidence and Next Step

Fresh_run: 36 fits, 144 fixed readouts, exact replay, source bootstrap. Cached_verified:
source assets, frozen forecasters, moment predictions and controls. Not_run:
independent selection/calibration/confirmation, new forecaster and deployment.
The cohort repeats 74 fit / 31 held source-qualified recordings across contexts/seeds.
The inner-held data were historically exposed; this is not full-pipeline OOF.

113 targeted tests pass; a separate arithmetic reporter checks 5,022 policy summaries
and 288 frozen controls. Reports and predictions replay exactly. Those checks
establish implementation consistency, not success of the scientific hypothesis.

Next, distinguish TRAIN budget/support infeasibility from optimization failure
using a fixed-action, per-record constrained oracle diagnostic. Keep every
recording, including unsupported ones. If feasible choices exist, test an
actual hard-decision constraint repair; if not, identify the missing outcomes
or causal context. Do not respond by increasing the budget, tuning held
thresholds, repeating the same meta-objective, or opening confirmation early.
That follow-up is not_run. Stage5C/SMC remain off; no metric/seconds, true-3D,
foundation, physical-safety or submission-readiness claim.
