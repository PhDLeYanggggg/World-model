# Failure Analysis

## Observed Failures Versus Hypotheses

| Issue | Evidence | Supported interpretation | Not established |
|---|---|---|---|
| Complete MLP gate fails | One log-loss and one overshoot-capture interval overlap zero | Consistency across all assignments is insufficient | No useful event signal exists |
| Added nonlinearity does not reliably help | All six full-input MLP-minus-linear log-loss intervals overlap; AP has a negative interval | Larger model is not justified by this comparison | All neural architectures are ineffective |
| Motion-only probability errors | Four MLP log-loss intervals favor linear and four Brier intervals favor prior | Ranking gains alone do not guarantee probability quality | Missing scene inputs alone caused the difference |
| Rare-event support | Three motion held views have zero positives; fitting motion localities can also lack events | Some ranking contrasts are not estimable | Zero events imply zero future risk |
| Teacher transport | Two-locality inner producer versus three-locality outer producer | Event threshold is producer-relative and can shift | The transport effect size is causally isolated |
| Severity and utility not tested | Binary event target only; no new policy evaluation | A learned event ranking is not an expected-cost repair | Forecasting, deployment or physical-safety improvement |

## Why Lower Training Loss Is Not Enough

All 72 full-input models in each arm reduce fixed-batch BCE. Full-input median
training loss changes from 0.22326 to 0.15053 for linear and to 0.11691 for MLP.
Motion-only MLP reduces loss in 70/72 views, compared with 53/72 for linear,
but held probability scores are often worse. These are dependent diagnostic
views and fixed-batch endpoints, not a generalization estimate.

This pattern is consistent with excess flexibility or locality shift. It
does not isolate overfitting, optimization error, or missing causal variables.
There was no post-readout tuning of width, training length, loss or threshold.

## Probability and Ranking Are Different Requirements

The full-input linear model has positive intervals against the fitting prior
for both proper scores and against all three ranking controls for AP.
However, tail overshoot capture is not uniformly supported. A model can
correctly rank event occurrence and still miss the few largest magnitudes.
This is the specific remaining severity problem, not evidence that the
classifier failed to learn anything.

MLP has a smaller descriptive median gap between mean probability and event
rate (0.00912 versus 0.01867), but that single marginal statistic does not
prove conditional calibration or dominance in proper scoring rules.

## Support and Dependence

Full-input held views have at least 12 positive recording-agent pairs and
three positive recordings. That is numerical support, not a power guarantee.
The same agents, recordings and localities recur across seeds and assignments.
The bootstrap unit is locality after averaging seeds, not individual windows.
Only four source localities contribute to each interval, and the source pool
has prior development exposure. No confirmatory population claim is made.

The unsupported motion ranking views are fold0/controller1/eu-locality-110
for seeds17/29/43. They have 2,221/1,934/2,033 known rows respectively but no
positive events. Omitting them would make the apparent consistency misleading.

## Limits on Causal Interpretation

This study changes the target, feature set and learning objective relative
to earlier any-harm, hurdle and moment studies. Their score differences do
not isolate the cap-event label as the sole cause of improvement. Likewise,
full versus motion-only changes the forecast pair and therefore the event.
A genuine scene/goal/interaction contribution requires a matched retraining
ablation with the same target population.

## Repair Direction

Keep the existing policy frozen. Register an auxiliary-label test that shares
causal representation with the joint H/H_E cost head, with a no-auxiliary
control at identical budget. Preserve learned nested cost constraints and
separate zero-reference protection. Event output must not substitute for
cost magnitude, and held outcomes must not choose its weight or threshold.
Any stacked prediction feature requires complete nested producer exclusion;
the existing inner-teacher lineage cannot automatically be reused one level
deeper. This repair is not yet trained or evaluated.

All diagnostics here are fresh post-readout descriptions, not a model search
or a claim that a causal failure mechanism has been proven. Independent
roles remain unopened; Stage5C and SMC remain disabled.
