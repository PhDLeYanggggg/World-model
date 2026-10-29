# Coupled Risk Failure Analysis

## Material Passport

Fixed-roster exposed-TRAIN fit. Descriptive mechanism comparisons, not causal
proof about real-world dynamics or an independent generalization test.

## Confirmed Repairs and Remaining Failure

The exact joint quadratic solution eliminates9,019 EH>H cases; all joint arms
also retain B+H<=E. Closed-form synthetic-grid tests and full real-data receipts
support that repair. However, joint-only easy MSE improves merely1.52%; the
predecessor's coherent weighted-harm arm also failed. Feasibility was not the
main predictive obstacle.

Recording-aware nested blending lowers easy MSE another24.95% relative to
joint-only fitting, but loses the original by5.69%. It loses the fixed mean-alpha
control by0.47%, although it beats shuffled alpha. This does not establish a
useful conditional-reliability contribution. Mean shrinkage explains much of
the gain against the poor unblended estimator; neither arm passes easy magnitude.

## Support Is Smaller Than Nominal Neighbors

Exactly3,183 queries have one reference recording among32 neighbors. They cannot
support the nested leave-record fit and return the original. Of62,902 occurrences,
22,032 have recording ESS<2. Median of view medians:rowESS7.78267 andrecordESS2.52932;
these are not pooled medians or independent sample sizes.

Prior inconsistency cross-tab:

| Recording ESS | Prior EH>H | Prior coherent | Total |
|---|---:|---:|---:|
| <2 | 4816 | 17216 | 22032 |
| >=2 | 4203 | 36667 | 40870 |

Most low-ESS queries are coherent, and46.60% of inconsistent queries have
ESS>=2. Also only2,062 inconsistent occurrences have rowESS<2. Concentration
and inconsistency overlap but neither identifies the other. No new cutoff is
selected from these readouts.

## Event Localization Remains Poor

Known easy events:3,762/61,335 repeated known queries. Primary easy-harm SSE
on events42.135673, non-events0.699167; original42.427698 and0.001031.
The modest event-error gain does not offset errors on non-events under the
registered equal-view metric. The primary puts97.37% of predicted easy-harm mass
on non-events. This is not a classification false-positive rate.

Predicted total easy harm24.190950 remains well below observed95.560642.
False-safe signed-score counts2645 versus original2825 and independent2444.
Lower MSE is not equivalent to a better2% safety certificate. All counts retain
unknown queries in coverage but exclude them from outcomes, never label them0.

Every source/context/head-seed mean loses original easy MSE. Locality020 remains
0.000114915 versus original0.000009287; locality0820.000059847 versus
0.0000000399. Tiny denominators make relative ratios unstable, so absolute
errors and source weights are retained. No source-specific rescue rule is fitted.

## Objective and Aggregation Must Not Be Confused

The blend minimizes the sum of all/easy normalized donor errors. All-risk errors
are much larger; a shared alpha can plausibly favor total harm at the expense
of easy harm. This is a NEW hypothesis, not yet a verified root cause: per-channel
nested-donor losses were not persisted in this experiment. Next collect those
components with exact control replay, then test a donor-easy-preserving fit if
the conflict is present. Do not infer safety from a smaller total donor loss.

Pooled occurrence all-risk SSE improves in several strata while equal-view all
MSE worsens1.35%. The denominators weight sources differently; this is an
aggregation reversal, not grounds to switch the registered primary metric.
Easy MSE remains worse in both recordESS<2 and>=2 strata; hard fallback on low
support alone would not establish that the remaining estimator is sufficient.

## Statistical Interpretation Audit:11/11 Considered

1. Aggregation reversal:equal-view versus pooled occurrence results differ; both denominators labeled.
2. Ecological inference:group summaries do not establish individual-safe interventions.
3. Selection bias:only the frozen selected query population is in scope.
4. Collider conditioning:policy/knownness conditioning prevents an unselected-population causal claim.
5. Base rates:3,762 easy events and61,335 known repeated occurrences reported;26 easy ranking views undefined.
6. Regression to mean:mean/shuffle controls distinguish some shrinkage effects; no pre-post causal claim.
7. Survivorship:all62,902 queries and all72 views retained,including1,567 unknowns and3,183 nested-unsupported rows.
8. Look-elsewhere:no best-arm substitution or confirmatory multiple-testing claim.
9. Forking paths:protocol/code committed before pilot; all previous failed experiments remain visible.
10. Causation:no real-world causal or physical-safety inference from fitting improvements.
11. Reverse direction:query future is scoring only; donor outcomes supervise fitting. Upstream exposure still prevents whole-pipeline OOF.

No p-values or independent CIs from these repeats. No holdout promotion or
scientific success claim. The original head's smaller fitting MSE is itself not
a new deployment certificate. Stage5C/SMC stay off.
