# Matched Dimensionless Risk Heads

This experiment isolates the representation of risk-head inputs. It does not
retrain a trajectory predictor, introduce a generative model, or calibrate a new
deployment policy. All four physical SDD sites have prior design exposure.

## Six Conditional Moments

For a fixed causal candidate trajectory and CV reference, let their observed
complete-path ADEs be `e_a` and `e_0`. Define benefit `B=max(e_0-e_a,0)` and harm
`H=max(e_a-e_0,0)`. The past-computable mean distance between the two forecasts is
`D`. The reverse triangle inequality gives `B+H <= D` on a common complete label
grid. This bound does not need a future endpoint at inference.

The easy indicator `E` is supervised by reference error and the frozen
training-only cutoff `c`. It is a label, never an inference input. One multi-output
forest predicts the conditional means of:

1. `B/D`;
2. `H/D`;
3. `E*H/D`;
4. `E*B/D`;
5. `E*e_0/c`;
6. `E`.

Rows without complete future labels have zero training weight. Zero-disagreement
rows also have zero fitting weight and cannot intervene. Output coherence is
checked after fitting and inference. Training MSE measures fractional target fit;
it is not held-source accuracy or calibrated risk coverage.

Crucially, overall expected gain and easy-weighted signed risk are different:
`g=D*(f_0-f_1)`, `r=D*(f_2-f_3)`, and the predicted easy reference-error denominator
is `u=c*f_4`. A head that predicts only easy benefit and easy harm cannot supply
the first quantity. The fresh native-feature control and dimensionless-feature
arm therefore both predict the same six targets.

## Feature Contrast and Fixed Budget

The native head consumes 356 existing features. The other arm retains the first
354 normalized features, removes native log scale, and replaces native log
forecast disagreement by `log1p(D/past_scale)`, yielding 355 features. Both forests
consider 118 columns per split. Source-only means and standard deviations are
fitted separately, with identical sample weights and 768,000 source draws.

All folds, predictors, forecasts, targets, seeds, tree counts, depth limits and
leaf-size settings are fixed. There are 72 fresh fits: four outer source sites,
three seeds, three candidate actions, and two feature representations. Outer-site
outcomes do not choose hyperparameters or thresholds. This source-excluded
construction does not erase historical exposure of those sites.

## Decision Rules

Intervention requires positive predicted overall gain, positive disagreement,
positive easy denominator and a non-static final history step. The registered
pointwise rule checks `r_i <= 0.02*u_i`. The population rule maximizes total
predicted gain in each simultaneous query subject to
`sum(selected r_i) <= 0.02*sum(all u_i)`.

A selected-denominator control instead checks
`sum(selected (max(r_i,0)-0.02*u_i)) <= 0`. Clipping the conditional mean signed
risk is not equivalent to estimating the expected positive harm; the latter is
represented by `D*f_2`. Neither rule is a distribution-free safety guarantee.

One additional contrast fixes the dimensionless policy's query intervention
count to the native population policy's count. Infeasible/support-deficient
queries fail closed. Such failures are retained, so the resulting aggregate
cannot automatically be described as a perfectly coverage-matched comparison.

Numerical feasibility is checked again in original units. Solver timeouts and
unproved optimality remain visible. No pairwise interaction consistency term is
introduced in this experiment.

## Statistical and Transfer Limits

All decisions are frozen before the new aggregate readout. Reports retain every
registered arm, available-label and complete-path metrics, zero-reference harms,
partial-future bounds, and per-site/per-seed results. The 3,000-resample paired
bootstrap has four physical sites, not hundreds of thousands of independent
windows. Its intervals are conditional development evidence, with no
multiple-comparison or independent-confirmation claim.

The reported gains use CV as reference; CV is not silently renamed the strongest
causal baseline. Protected damping is a separate matched action control. Gains
over CV alone do not establish a neural or multimodal contribution.

Removing two explicit native-unit inputs does not make the complete controller
unit-invariant. The native easy cutoff and original prefix tolerance still remain.
The separate quantized-prefix precision repair is deliberately not integrated
here, to avoid changing the forecasts in a head-feature comparison. Its successful
finite-factor numerical probe is not predictive transfer evidence. No IMPTC or
DroneCrowd future outcome is evaluated, and no deployment change follows from
this source-only readout.
