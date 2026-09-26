# Failure Analysis

## Observed Failure

The registered context probe does not consistently reduce transported
easy-harm MSE. Its full-input gains against the original are small and
assignment-dependent; motion-only points are all negative. Tail/coverage
guard completion does not cancel the failed primary criterion. The original
cost model, not an already worse auxiliary model, remains the primary control.

## What the Evidence Supports

1. A global intercept shift is insufficient: one of six full-input MSE
   intervals is positive. Adding the seven contexts yields two, not six.
2. Some fitting-defined relative bias directions repeat across localities,
   especially speed change and closing speed in the full-input estimator.
   That does not establish transport of the required correction magnitude.
3. Event ranking and magnitude remain distinct. Four full-input AUROC
   intervals improve, while the MSE criterion fails. A higher event AUROC
   is not evidence that a policy can estimate the expected cost of switching.
4. Context is not uniformly useful: direction reversals persist, and the
   motion-only result does not support a general context repair.

## Unresolved Mechanisms

| Hypothesis | Current evidence and limit |
|---|---|
| In-sample residuals differ from deployment residuals | Known design limitation: the base estimator saw the residual-fitting rows. This round does not identify its causal contribution. |
| Context directions transfer but magnitudes do not | Compatible with the sign tables and failed MSE gate; not a controlled causal attribution. |
| Additive tercile bins miss nonlinear interactions | Plausible but untested. No post-readout bin, ridge or interaction sweep was run. |
| The frozen all-harm upper bound limits the correction | H_E is clipped to [0,H] by design. A cap-limitation decomposition was not run; do not claim it is the root cause. |
| Support and source shift limit transport | Each assignment has four held-development localities and dependent windows. Neither this study nor the preceding coarse radial proxy rules out conditional shift. |
| Implementation or sampling error | Coefficients, predictions, labels and numerical metrics are subject to full replay. Passing these checks cannot make the scientific result positive. |

## Next Repair Constraint

Change residual provenance before increasing probe flexibility: evaluate a
properly nested fitting-locality-OOF construction against this in-sample
control. Audit the easy-label rule, normalization, encoder/head fitting and
every producer exclusion in each inner fold. Each inner residual must use a
predictor not trained on that residual's labels, and the outer held locality
must be absent from the complete chain. Existing outer-fold models cannot
be swapped in when they have seen the current outer-held locality.

Keep all seven summaries and the original strong comparator rather than
selecting closing speed from a favorable held table. Any changed target rule
must be made explicit before training; it cannot silently replace the prior
endpoint. Preserve all adverse assignments and independent reserved roles.

No trajectory training, new deployment, threshold tuning, Stage5C execution
or SMC occurred. Coordinates and annotation steps remain uncalibrated for
physical metric/time claims; automatic labels are not human gold.
