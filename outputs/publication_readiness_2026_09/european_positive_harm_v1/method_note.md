# What The Positive Link Does And Does Not Establish

For each frozen tree leaf and each of total/easy harm, let `mu` be its original
weighted training mean, and `q` the seven past features after training-only
standardization and clipping. Center `q` within the training leaf. The new raw
harm estimate is

`h(q) = mu * exp(beta' q) / E_train[exp(beta' q)]`.

The denominator is stored from training. It is not recomputed on validation,
selected rows, a new scene, or a batch at inference. A zero training-harm mean
leaves that channel unchanged. The model does not claim that such leaves have
evidence of zero population harm.

For positive `mu`, the fitted objective, relative to its zero-slope value, is

`log E_train[exp(beta' q)] - E_train[y*q]/E_train[y] * beta + ||beta||^2/2`.

This is a mean-normalized conditional Poisson deviance objective with fixed
regularization. It does not assume integer-valued harm observations or make
a Poisson confidence claim. The fitted objective and the additive control's
squared-error objective are different; their training loss numbers must not
be compared as the same quantity. Downstream validation readouts are common.

Its Hessian is the exponentially tilted feature covariance plus the identity.
The independent small-data test compares the batched Newton fit with SciPy
BFGS. Real runs also verify training mean preservation, convergence, exact
refit, raw non-harm component preservation, and serialized inference replay.

## Boundaries

- Positivity prevents one particular clipping mechanism. It does not make the
  estimate large enough, establish calibration, or resolve unseen support.
- Mean preservation holds in each TRAIN leaf before the common feasibility
  projection. It need not hold on validation or after that projection.
- Projection can change benefit when harm changes. Raw B/R/ER are frozen;
  projected benefit is not falsely described as frozen.
- A small mean loss does not establish low selected risk. Selection depends
  on these same predictions, and both harm and reference errors matter.
- Whole-easy net degradation is not selected positive-harm/reference risk.
- Unknown future outcomes remain unknown in offline completion bounds. Their
  potential harm is not zeroed, inferred as observed, or used as a quality gate.
- This experiment learns decision costs for frozen trajectory candidates. It
  is not a new neural world-dynamics architecture or independent safety result.

The fixed numerical exponent guard is only overflow protection; its activation
count is reported. No calibration threshold or risk budget is changed.
