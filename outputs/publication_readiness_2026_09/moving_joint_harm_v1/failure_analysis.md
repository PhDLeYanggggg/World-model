# Why Joint Event Supervision Did Not Repair Admission

The full registered training, validation and exact replay are complete. This
is a negative model experiment under unchanged rules, not an aborted run or
an identity-fallback certification. No new H evaluation was used.

## What Improved, And What Did Not

On all legal actions with available outcome labels, event supervision reduces
harm envelope-fraction MSE by 0.05091 [0.03162,0.07210] and easy-harm MSE by
0.05191 [0.03233,0.07457] relative to the equal-architecture product-moment
control. Reference normalized MSE also improves. These are real diagnostic
improvements, not passed policy gates.

The product model's probability factors have no occurrence supervision, so
beating their Brier scores is not enough. The event model loses to fitting-only
frequency priors: H Brier gain -0.08270 [-0.13022,-0.04918], EH gain -0.09111
[-0.16280,-0.04014]. This supports a generalization problem in event prediction;
it does not establish that capacity, class imbalance or overconfidence is the
unique cause. No calibration guarantee follows from cross entropy alone.

## Selection Changes The Error Regime

All ratios below compare realized and predicted masses on identical known
rows for that region. They pool repeated rows across contexts and are
descriptive, not source-level confidence bounds.

|Event model region|Actual/predicted H|Actual/predicted EH|
|---|---:|---:|
|Frozen proposer choices before veto|0.4602|0.1520|
|Event-veto admitted subset|5.9753|162.4000|

The model is conservative over the first pool, but optimistic on the retained
subset. Global risk inflation or average MSE improvement is therefore not
sufficient evidence of selected-set reliability. The selected subset differs
by construction; these ratios do not identify a pure causal effect of vetoing.
The complete fixed-proposal source-macro diagnostic is unavailable when a
required view has no supported proposal; it is not replaced by pooled ratios.

## The Actual Decision Still Fails

|Issue|Measured evidence|
|---|---|
|Complete nonempty primary unavailable|Event 24/276 source views; product 26/276; no dropped views|
|Event veto loses to matched utility ranking|Known -1.031pp [-1.886,-0.369]; direct lower -1.543pp [-2.811,-0.601]|
|Auxiliary loss does not establish full-policy improvement|Event minus product known +0.275pp [-0.062,+0.665]; lower +0.099pp [-0.243,+0.446]|
|Original full risk fails on estimable cases too|Event 382 known all-risk and 399 known easy-risk violations; only 47/696 supported|
|Support is independently incomplete|213 undefined selected easy-risk views; 101 no-intervention views; 30 undefined full known easy-net views; counts overlap|
|Worst easy case remains unacceptable|247.37% worst defined signed recording easy-net degradation, not a dataset mean|

Event source-macro switch rate is 18.685%, versus 18.015% for product. Each arm
has its own control matched by query count and amplitude histogram. Comparing
the two full strategies is a secondary contrast with different coverage, not
a matched-ranking claim. Displacement magnitudes are not matched.

Falling fitting loss, exact replay and a positive point estimate against the
other head cannot override the negative matched comparison or original safety
requirements. Likewise, a recording without an easy reference cannot be made
estimable by better training. The user explicitly retained that rule.

## Next Model Work

1. Complete a task-matched public cost-deferral comparator. Its logistic
   surrogate is implemented and equation-tested, but real fitting is not_run.
   Keep the same forecast bank, roles, original risk readout and all views.
2. Test selection-conditioned supervision rather than uniform all-action
   average loss. Construct training selection masks only with source-excluded
   proposers; no held validation sources may enter their fitting. Compare a
   fixed fitting-prior shrinkage control and the selected-tail loss before
   claiming calibrated confidence. Register the next comparison before scoring.
3. Treat a successful development repair and independent confirmation as
   separate requirements. Reserved source roles stay closed. Neither more
   resampling of these sources nor moving an old score to a new file creates
   independent evidence.

These are next experiments, not completed improvements or a reason to relax
the 2% limit. Current deployment is unchanged. No Stage5C or SMC. Native 8/12
annotation-order pixels remain non-metric and not verified seconds; this
cost-head experiment is not true 3D or foundation world-model success.
