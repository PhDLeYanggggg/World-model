# Failure Taxonomy

## Supported Observations
1. Extra sequence features do not reliably outperform the seven-summary
   control. Full history-neighbor versus old-summary primary intervals are
   0 positive / 2 negative / 4 overlapping; motion-only is 1 / 5 / 0.
2. Fitting improvements often do not transport even to an inner fitting
   locality. Against old summaries, history-neighbor improves the descriptive
   full-input fitting MSE in 150/216 views, but 82 of those improvements are
   fitting-only. Motion-only has 162 fitting improvements and 124 fitting-only.
3. The added neighbor block is not consistently helpful: full primary has
   three negative intervals versus history alone. This is a negative result,
   not evidence of a useful interaction module.
4. The frozen predicted all-harm cap is too low for some realized easy-harm
   labels. Even a label-aware best value inside that cap leaves a median
   65.18% of the full raw MSE. The model never receives that label-aware value.
5. Event mass is concentrated. Motion-only has 158/216 dependent views below
   ten effective event tracks, including three zero-event views. Undefined
   coverage/tail quantities remain missing. Repeated windows cannot fix this.

The ceiling calculation uses realized costs, whereas the model predicts an
expected cost. Even an unbiased expected all-harm prediction is not an upper
confidence bound on every realized event. A large pointwise ceiling floor
therefore does not prove bias in the predicted conditional mean, nor does it
distinguish stochastic label variation from a harmful modeling constraint.
That distinction needs the next matched causal attribution, not a label oracle.

A synthetic arithmetic check makes the distinction explicit. Let realized
harm be 10 once and 0 nineteen times, with both predicted means fixed at 0.5.
The predicted and realized means agree exactly. Yet MSE is 4.75 and the
pointwise ceiling floor is 4.5125, or 95% of that MSE. This is a mathematical
counterexample to the bias interpretation, not an additional real-data result.

## Plausible Mechanisms, Not Identified Causes
High-dimensional linear features fitted on two localities can overfit or
transport poorly. Single-locality nuisance heads and two-locality scoring
heads differ in fitting size, normalization and easy cut. The linear readout
and selected complete-history neighbors can miss nonlinear intent and agents
outside the chosen neighborhood. Pixel viewpoint and detector-box jitter can
also obscure motion. This experiment does not separately identify these causes.

## What the Negative Result Does Not Establish
It does not show that all history is uninformative, that all neural dynamics
are impossible, or that extra recordings alone would solve the problem. It
does not prove that the cap is the sole cause. The median remaining fraction
is substantial and the cap diagnostic uses realized labels. Raising a cap is
not justified as a deployment change by this lower bound.

The full-input median easy-harm MSE is 0.04754 raw, 0.04686 score-only,
0.04577 old-summary and 0.04605 history-neighbor. These dependent-view medians
are not paired effect estimates. The paired intervals remain the registered
screen, including negative and missing results. Full coverage near one in a
median view coexists with a maximum 28.10 for history-neighbor; a favorable
aggregate mean cannot certify subgroup risk.

## Consequence
Do not enlarge the network, tune ridge/thresholds or promote a favorable
motion-only assignment. First separate the frozen-cap limitation from the
prediction information in the existing fitted readouts, still on fitting
development data and against all matched controls. If that attribution is
negative, prioritize better causal observations and event-bearing training
localities instead of another readout sweep. Independent calibration and
confirmation are not repair-selection resources.
