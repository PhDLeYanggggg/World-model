# What Boundary Supervision Can And Cannot Establish

This is a controlled repair of the current cost guard, not a new deferral
framework, a calibrated risk estimator or a trajectory contribution. The model
and objective were registered before training. This note explains their scope;
it changes no training, evaluation, threshold or advancement condition.

## Different From Previous Signed-Error Fits

The earlier signed-risk auxiliary minimizes squared error on H-0.02R (and its
easy counterpart), including a 4x underprediction arm. Large residuals receive
large gradients even if their sign is already unambiguously unsafe. The new
auxiliary instead penalizes both directions of boundary-event disagreement
through a cost-tied log ratio. It adds no independent classifier parameters.
It is not a one-hot label for the best baseline, and does not modify ranking.
Earlier native geometric and protected-risk controls also used BCE, for an exact
zero-reference event or a separate harm-event head. BCE itself is therefore not
new to this project. This comparison tests its tied use at the deployed all/easy
budget boundary in the current coherent critic, with the ranker held fixed.

The reference branch retains its original regression loss. Its value is detached
inside this auxiliary, preventing a direct gradient that wins only by increasing
the budget denominator. The shared encoder still affects references indirectly;
the control must therefore inspect the complete predictions and actual risk.

## Event Confidence Is Not An Expected-Harm Budget

Let reference R be fixed at1. Suppose H=0 with probability0.99 and H=1 with
probability0.01. The probability of violating H<=0.02R is0.01, while expected
harm is0.01 and meets the mean budget. Changing only the rare harm from1 to100
leaves that event probability at0.01 but changes expected harm to1, far above
the same budget. Event classification alone cannot distinguish their severities.

The present cost decoder ties the logit to H/(0.02R), so even a perfectly learned
event probability would not identify the conditional expected harm. This is why
the original moment losses remain, why outputs are called risk-oriented proxies,
and why the registered full-policy recording criterion is indispensable.
Adding the BCE may help realized decision signs or may conflict with severity
learning; neither outcome can be inferred from an auxiliary-loss curve.

For a fixed positive budget c, ignore the numerical epsilon and write the tied
event probability as p(h)=h/(h+c), with q=P(H>c). Its population BCE derivative
is (p(h)-q)/h. At the conditional mean mu=E[H], the magnitude-MSE derivative
is zero but the auxiliary derivative generally is not. They agree only in the
special case mu/(mu+c)=q. Thus even unlimited capacity and perfect optimization
do not generally make this tied combined objective a consistent conditional-
mean cost estimator. Epsilon regularizes the formula, not this interpretation.
This elementary single-output argument is not a new theorem or a unique causal
attribution of every empirical failure in the shared finite network.

A future repair should separate event prediction from cost magnitude semantics
or use a proper conditional-moment factorization. Existing separate-event and
hurdle assets must be checked before another fit. Simply training this same
inconsistent output tie longer or calling its sigmoid a calibrated budget is
not a supported solution.

The 2% criterion on the realized full recording also does not follow from
per-action conditional mean feasibility, even if those conditional means were
correct. Missing-outcome completion accounting and selected-reference support
are further requirements, not properties a classifier can remove.

## What Would Count As Progress

Retain both full-budget arms and all original contexts/recordings. A lower loss,
more selected easy anchors or a better observed average alone is insufficient.
Report paired full completion-lower utility, every required recording's all/easy
risk, easy degradation and two-sided guard errors together. An observed failed
control stays failed. Future-label safe anchors are diagnoses, never inference
inputs or an oracle deployment policy.

All evidence remains exposed EuropeanSquares SOURCE fitting, obs8/pred12/stride12
raw frames, detector-silver/oblique local-image. No independent, metric/seconds,
human-gold, true3D, foundation or physical-safety statement follows. Stage5C and
SMC remain off. If this repair fails, do not merely repeat a threshold sweep or
claim that the current schema is irreducibly insufficient without further evidence.
