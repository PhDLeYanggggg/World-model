# What The Composed Loss Fixes, And What It Does Not

## Result

The original rules are retained. There is no identity-fallback certification,
threshold search or new reserved/H readout. The model is not deployed and the
project is not submission-ready.

This experiment isolates one loss change, not a larger network or new data.
All three freshly trained independent controls exactly reproduce the previous
separate64 models. The added loss directly trains the per-channel conjunction
of risk and decision signs while preserving the original objectives and
conservative supervision on unknown outcomes.

## Fitting Improvement Is Real But Limited

| Context | Composed all-cost reduction versus independent | Easy-cost reduction | Risk MSE increase |
|---|---:|---:|---:|
| critic0 | 30.942% | 13.644% | 1.028% |
| critic1 | 14.679% | 9.457% | 0.591% |
| critic2 | 9.890% | 1.186% | 0.157% |

All three reduce fitting MSE versus their warm initialization and improve both
observed decision-cost channels. The previously failed internal screen now
passes3/3. In critic0, all-cost changes from0.004702 at initialization to0.004896
with independent losses, versus0.003381 with the composed loss. This supports
the specific fitting hypothesis: independently trained branches did not optimize
their combined decision well. It does not show universal superiority: regression
MSE is slightly worse than the matched independent arm in every context.

Error attribution also confirms that critic0's all false-reject cost falls from
0.00470249 under independent training to0.00324404 under composed training;
its false-safe cost falls from0.00019369 to0.00013719. SOURCE048, which dominated
the previous false-rejection increase, improves on both channels.

The three-context fitting screen is not an every-source success statement.
Within critic1, SOURCE053 and065 easy decision costs still rise170.74% and
107.84% versus warm initialization, while SOURCE076 falls82.79% and dominates
the pooled improvement. These are small normalized cost components, not easy
trajectory-degradation percentages; the underlying absolute costs remain in
the attribution JSON. This residual source imbalance is not hidden by3/3.

## Held Development Does Not Establish Safe Generalization

| Frozen policy | Known ADE gain over floor | Hard gain | Switch rate | Worst defined easy-net degradation | Full original risk/support |
|---|---:|---:|---:|---:|---:|
| Original constant control | 3.601% | 4.626% | 15.014% | 38.905% | 0/17 |
| Independent two-head | 0.745% | 0.995% | 5.231% | 8.116% | 0/17 |
| Composed two-head | 0.759% | 1.009% | 5.294% | 8.038% | 0/17 |
| Composed risk-only diagnostic | 3.315% | 4.287% | 14.596% | 40.378% | 0/17 |

These are source-macro results on seven already exposed SOURCE units and17
recording views, with partial observed outcomes. They are not complete-roster
12-step ADE, independent test evidence, or statistical significance claims.
The constant control is not safe either; greater utility does not make it
deployable under this contract.

The primary matched contrast is only+0.01313 percentage points of known ADE
gain. Missing-label completion bounds are[+0.00766,+0.01860]pp. These bounds
are NOT sampling confidence intervals. The contrast against the original
constant control is-2.84203pp, with completion bounds[-4.03809,-1.64596]pp.
This is not a successful main-method improvement.

Three views have undefined selected-easy risk. Ten views violate the known
all-risk limit and12 violate the known easy-risk limit. The original complete
risk/support criterion passes in0of17 views. Empty controls and missing support
remain NA; no fallback exception converts them into a pass.

The worst easy-net slice is SOURCE062/record27:8.0378% known easy-net degradation,
17.4194% known all positive-harm risk, and79.3472% known easy positive-harm risk.
Records30/26/31 of the same source have easy-net degradation6.8573%/4.3628%/4.2892%.
Positive-harm risk and signed easy-net degradation are different quantities;
both remain in the report rather than being substituted for each other.

## Failure Taxonomy

1. **Composition mismatch on fitting data: partly repaired.** Explicitly
   training the maximum improves the combined observed sign decisions in all
   three contexts, unlike the previous independently trained branches.
2. **Fitting-to-held transfer: unresolved.** Large fitting cost reductions yield
   only+0.013pp descriptive held gain. The result does not identify model size,
   missing labels or any single feature as the unique cause.
3. **Utility versus rejection: unresolved.** Adding the decision gate to the
   frozen composed risk head loses2.556pp of known gain while reducing worst
   easy-net harm. Removing that gate restores utility but raises worst easy-net
   degradation to40.38%; it is not a valid deployment repair.
4. **Objective versus deployment mismatch: still present.** Training averages
   legal candidate actions and separate all/easy channels. Deployment uses the
   frozen router's chosen action, both channels, both heads and support checks.
   The new surrogate trains only per-channel conjunction, not the complete
   selected-policy risk ratio. This is a code-supported remaining mismatch,
   not a proven exclusive cause of the held failure.
5. **Evidence support: independently unresolved.** The original24-group primary
   remains not estimable. Neural weights cannot create missing reference events.
   Keeping the original rules means leaving this limitation visible.

## Next Decisive Work

Keep the model and thresholds frozen. On the permitted fitting roles, separate
errors on router-proposed actions from errors on other legal actions, including
false-safe, false-reject and source-level costs. This will test whether the
remaining training/deployment mismatch explains where the improvement is lost.
Any subsequent loss change needs a fixed matched contrast and the same original
rules, not a posthoc threshold chosen from SOURCE062 or a removed safety branch.

Independent supported confirmation, three-seed main experiments and matched
published controls remain separate requirements. This diagnostic does not
establish a new forecasting contribution, formal risk guarantee, or novelty
of cost-sensitive deferral. Native obs8/pred12 image coordinates only; no metric,
seconds, true3D, foundation or human-gold claim. Stage5C and SMC remain disabled.
