# Net Easy Risk Recovers Utility, but Does Not Establish Safe Deployment

## What Was Run

Fresh fitting: 36 four-output risk forests, 128 trees each, four excluded source
sites and three seeds. The native arm64 CPU4/workers0 pilot resumed to the entire
fixed budget. Fitting-loop time totals 918.990 seconds; this excludes loading,
decision allocation and evaluation. Training sampled no unknown-label rows.

Forecasts, old neural cost heads, source preprocessing, sampling counts and cost
labels are cached-verified. No Transformer/EqMotion forecast model was retrained.
All nine registered policies cover 175,756 past-eligible windows and 188,388
query/action/seed instances. The protocol is obs8/pred12 native annotation steps,
SDD stride 12 and annotation pixels, NOT the historical raw-frame t+50 protocol.
Seed-averaged errors are not a prediction ensemble.

## Main Readout

Equal-site available-point ADE gain over CV, with worst positive-easy degradation
over individual site/seed views:

| Frozen predictor | Old strict ADE gain % | Net-population ADE gain % | Hard gain % | Worst easy degradation % | Zero-CV harmed query/seed instances |
|---|---:|---:|---:|---:|---:|
| Damping005 | 3.6347 | 1.0945 | 0.8526 | 0.0000 | 0 |
| Transformer | 2.4368 | 2.9392 | 2.4349 | 0.8636 | 3 |
| EqMotion | 1.6093 | 2.8268 | 1.6807 | 1.8384 | 6 |

The Transformer difference versus its old strict rule is **+0.5024 percentage
points**, paired site-bootstrap CI95 **[-1.2057, +2.2104]**. It is not a stable
superiority result. EqMotion's corresponding difference is **+1.2175 pp**,
nominal CI95 **[+0.1528, +2.5668]**. Damping loses **2.5402 pp**, CI95
[-3.7440, -1.4836]. Its old strict rule also exceeded the 2% easy ceiling, so the
tradeoff differs from the neural predictors. No predictor is silently dropped.

EqMotion's gain comes with worse easy results: worst degradation rises from
0.4522% to 1.8384%, and its mean positive-easy gain drops 1.1049 pp relative to old
strict. Both neural net-population policies harm zero-CV instances whereas their
old strict policies harm none. Passing an aggregate 2% ceiling is not individual
safety, a zero-harm guarantee or authorization to deploy.

## What the Controls Explain

Using the SAME new forest, changing positive-population to net-population adds
0.3562/1.8963/2.0363 pp ADE gain for damping/Transformer/EqMotion. Their nominal
paired intervals are positive. The same-head pointwise contrast is also positive.
This supports the development hypothesis that positive-only harm was overly
restrictive for the net-degradation objective.

After matching intervention counts, the gains shrink to 0.1061/0.4167/0.0985 pp.
Thus much of the population improvement disappears when coverage is controlled;
it cannot all be attributed to better ranking. The Transformer count-matched
control has one explicitly failed query, so it is not perfectly matched globally.
The sensitivity bound for that query is below 0.001 pp primary gain, much smaller
than the 0.4167 pp contrast, without substituting an outcome-selected repair.

The new forest with the OLD positive accounting performs worse than the previous
forest's positive accounting for all three actions. Simply adding a benefit
target or reporting a lower mean training loss is not the improvement claim.
Full tables retain the effectively inert positive-pointwise controls as negatives.

## Remaining Failures

1. No independent calibration or confirmation. These are four development-exposed
   physical sites. Excluding a site during a fit does not undo design exposure.
2. The signed constraint is weaker than positive harm: improvements can offset
   harm. It does not prove a better positive-risk guarantee.
3. Zero-CV harm persists. Relative degradation is undefined at zero reference;
   these cases must not disappear from evaluation.
4. The effect is site-dependent. Transformer loses to old strict at deathCircle
   and gates while gaining at coupa and hyang. EqMotion's largest gain is coupa;
   its gates difference is only 0.0005 pp.
5. Training risk labels require complete futures while primary ADE permits
   available future points. Missing support and conditional-risk calibration are
   still unresolved. Unknown/incomplete decisions and partial-label bounds remain
   in the full analysis; missing labels are not known negatives.
6. Three strict primal/dual checks fail closed. No new budget-constrained action
   violates its predicted budget, but a count-matched query is not optimal/matched. See
   the [numerical sensitivity record](numerical_sensitivity.md).
7. No new geometry term or interaction mechanism is tested here. The previous
   negligible nonadditive-interaction finding remains negative. This is not a
   new neural dynamics, JEPA, multimodal contribution or world-model result.

## Verdict

The accounting hypothesis has useful **development evidence**, particularly for
EqMotion, but this does not establish a new best deployable policy. No model is
promoted; no threshold is selected from these outcomes. All 27 predictor/policy
rows and adverse results remain available. Nominal 3000-resample physical-site
intervals are exploratory, not population or multiple-comparison guarantees.

Fresh_run: risk fitting, causal decisions, aggregate readout and diagnostic
sensitivity. Cached_verified: forecasts, source cohort, preprocessing and old
cost labels. Not_run: independent calibration/confirmation, new external
prediction, new forecast training, Stage5C and SMC. DroneCrowd stays closed;
DUT remains exposed diagnostic and HT21/CroHD remains quarantined.

## Reproduction and Evidence

Full decision and aggregate replay are exact. Separate arithmetic verifies
565,164 constraints, 882 small-query optima, 1,620 scene reductions and 54 paired
contrasts; three larger exhaustive checks are explicitly skipped. The three
nonoptimal fail-closed queries and one count mismatch remain recorded. All 144
scoped tests pass; the unrelated legacy suite was not rerun. These checks verify
the computation, not independent generalization or safety.

See the [full table](results.md), [training losses](training_losses.md),
[all-control figure](risk_tradeoff.svg), [per-site/seed results](site_seed_results.csv)
and [execution record](execution_notes.md). All required processes completed.

## Next Repair

The next repair should target causal support for zero-CV-sensitive interventions
and site-dependent risk error, while securing admissible independent calibration
data. It should not turn these same development outcomes into a new threshold
search or claim that risk-budget feasibility is real-world safety.
