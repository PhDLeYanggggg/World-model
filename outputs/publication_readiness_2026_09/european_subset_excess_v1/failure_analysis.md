# Failure Analysis

## Observed, Not Inferred

Both arms reduce their own fixed fitting-monitor loss in108/108heads.
Optimization executed successfully; a lower training loss did not demonstrate
the requested held-source decision gain. Matched-count ADE advantage is not
supported, while the full-reference harm contrast favors the pointwise control.
Joint intervention counts differ, so joint differences alone cannot isolate
risk ranking quality.

Descriptive changes in held MSE under aggregate versus pointwise subset loss:
controller-admission all-risk MSE -3.49%; low-disagreement all-risk MSE -1.21%;
whole-query all-risk MSE -0.62%. In contrast, high-disagreement all-risk MSE
increases0.44%, whole-query easy-risk MSE increases1.64%, and low-disagreement
easy-risk MSE increases1.62%. These are post-readout descriptive ratios of fixed
summary point estimates, not extra registered significance tests.

The frozen controller subset is nonempty with known labels in only37.74% of
held queries on average. Low/high disagreement subsets have99.77%/69.92%
coverage;29.96%of held known queries are singletons. Thus the training proxies
do not represent every optimizer-selected combination. We did not establish
that coverage alone causes the failure, or that different proxies will fix it.

## Taxonomy

| Issue | Evidence | What remains unproved |
|---|---|---|
| Optimization failure | Both arms' own monitored fitting losses decline108/108 | Global optimum or sufficiency of2000updates is not proved |
| Loss-to-decision mismatch | Some held group MSEs improve but count-matched harm increases | The precise contribution of proxy mismatch versus estimator bias needs a controlled test |
| Aggregation ambiguity | The shared pointwise anchor prevents complete cancellation in a unit test | It does not identify separate cost moments or guarantee useful conditional ordering |
| Source shift | All models are source-separated; only5/12rank contrasts positive | This experiment cannot distinguish missing causal cues from limited training-source diversity |
| Safety failure |84risk-violating and16undefined joint views; easy net error remains preserved | Easy preservation is not selected-harm calibration |
| Measurement defect | Ten inherited rank views abstain and have undefined selected ratios | A fixed-denominator diagnostic does not repair the old primary retrospectively |
| Tiny secondary signal |0.004553%rank advantage over cached control, nominal CI barely positive | No useful deployment gain or isolated aggregation effect |

## Next Controlled Action

Do not run another unregistered loss-weight or threshold sweep on these readouts.
First quantify, from the frozen actions, signed-risk residuals specifically on
the optimizer-selected sets and compare them with the fixed training proxies.
Separate lost benefit from added harm, and uncertainty/support from conditional
bias. Keep every locality, unknown label and abstention. This is a diagnosis,
not a new model-success claim.

Then preregister one targeted change, for example source-excluded action-set
supervision with explicit estimator uncertainty, if that diagnosis supports it.
Retain frozen predictors, utility and matched-count controls. Define any future
coverage-aware risk contract before fitting; never recycle the new diagnostic
as the old2%certificate. Independent calibration/confirmation stay closed until
the development method and claim are ready. Public-method comparisons and
independent generalization remain unfinished.

This study is image-local detector-silver development evidence on raw frames,
not metric/seconds, physical safety, true3D or foundation success. Stage5C/SMC
remain disabled. The deployment policy is unchanged.
