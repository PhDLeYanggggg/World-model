# Failure Taxonomy: Attribution Controls

## 1. Coverage Is Not Ranking

Full neural vs ridge has three positive all-ADE intervals, but two negative and
one overlapping. On exactly the same forecasts and matched per-query counts,
there are no positive all-ADE intervals; two are negative. Motion-only neural
ranking loses to ridge in all six seed-mean all-ADE comparisons. Thus a larger
intervention set or different risk scale can explain an apparent average gain
without better within-query ranking. This is supported diagnostic evidence,
not proof that coverage is the only mechanism.

Matched controls are nontrivial: full pairs change228-2,754 row choices in112-
1,122 queries between rankings. Motion-only pairs change122-1,708 rows in61-735
queries. Counts range1,738-13,388 and5,763-12,046 respectively. Many queries have
K=0:3,119-5,679 full and3,274-4,469 motion-only, out of7,087. This limits the
matched experiment to common supported coverage; it does not evaluate the value
of every intervention the less-conservative system would have made.

## 2. Neural Trajectory Necessity Is Subset-Specific

Removing neural forecasts and both neural policy bits, then retraining cost
heads, retains most overall gain. Full neural vs motion-only has only two
positive all-ADE intervals, four overlaps, and no positive hard-ADE interval.
There is a real narrow positive result: all six easy-subset intervals favor full
neural, by0.778339%-4.293073% in seed means. Do not hide this, but do not generalize
it into broad dynamics superiority. Both systems keep easy degradation below2%
against CV, yet comparisons against each other can differ by locality and event.

Motion-only still contains learned motion-floor scoring and newly trained
neural cost scoring. It removes neural trajectories, not all neural computation.
Full-vs-motion changes both reference and candidate endpoints, so its attribution
is at system level, not an isolated coefficient or perfectly matched forecaster.

## 3. Utility Refinement Is Not the Main Observed All-ADE Effect

Keeping neural risk but replacing neural utility with ridge changes full all-ADE
by only-0.003642% to+0.077216% in seed means (3 positive,1 negative,2 overlapping
intervals). Replacing only risk yields-0.145889% to+3.091746% (3 positive,2
negative,1 overlap). This points toward risk scaling and support as more relevant
than utility-model complexity, without isolating loss design from architecture.

## 4. Net Easy Preservation Is Not a Risk Certificate

The full neural system preserves net easy error in18/18 configurations with a
worst-locality degradation0.730345%. Nevertheless positive-harm accounting on the
same event violates its0.02 ratio in68/108 dependent locality/views. Ridge does
not solve this automatically: full ridge violates41/108 and motion-only neural
38/108. Summed benefit can offset harm in net ADE while individual worsening
remains substantial. No independent risk calibration was performed.

## 5. Training and Support Checks Are Not the Blocker Here

All36 new heads completed72,000 updates, with no unknown-label sampling and
matched parent training draws/support/CV scale. All18 risk heads reduced fixed
training-batch loss. This rules out an unexecuted training pipeline as the
explanation, not underfitting, distribution shift, label noise or loss mismatch.
Detector-derived trajectories and six development localities still constrain
generalization. More epochs are not justified solely because matched ranking
fails; a source-only diagnostic should identify what the current loss misorders.

## Repair Priority

1. Preserve ridge and motion-only controls; do not remove an inconvenient baseline.
2. Analyze event-conditioned moment calibration and support on exposed data;
   separate intervention coverage from ranking and quantify finite-scene power.
3. Register independent calibration once the risk estimand and fixed family can
   be evaluated without changing the approved2% tolerance. If evidence is too
   weak, report abstention or a statistical limitation rather than a guarantee.

No test-driven threshold search, new deployment, Stage5C or SMC. Native image
pixels and annotation steps only; no true3D/foundation/metric/seconds claim.
