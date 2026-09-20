# Global Harm And Conditional Relative Easy Risk

This is an explanation of the existing negative result, not a new theorem,
calibrated deployment rule or primary-metric amendment.

## Existing Quantities

For agent i, let B be the fixed baseline forecast, N the fixed neural forecast,
a_i the past-input selection decision, and L the unchanged normalized ADE.
Define d_i = L(N_i,Y_i)-L(B_i,Y_i) and h_i=max(d_i,0). The policy constrains the
predicted mean of a_i*h_i over all n past-eligible query agents. The conservative
and moderate global budgets are 0.01 and 0.03. Benefits do not offset positive
harm in this constraint. These are expected costs, not failure probabilities.

The original easy criterion instead uses w_i=1{L(B_i,Y_i)<=tau} and, on the
fixed complete-label evaluation cohort, checks

```text
sum_i w_i * a_i * d_i <= rho * sum_i w_i * L(B_i,Y_i),   rho = 0.02.
```

For a positive denominator, this is exactly the existing 2% relative aggregate
degradation constraint. Zero baseline denominator makes the percentage undefined;
the diagnostic must retain that fact rather than inventing a percentage.
Easy membership is outcome-defined and only a training/evaluation label. It
cannot be passed to the deployed selector as a known Boolean.

## Nonimplication

Consider 100 agents, one easy agent with baseline ADE 0.001 and selected ADE
0.1, and 99 unchanged non-easy agents. A truthful global mean positive harm is
0.099/100=0.00099, below the 0.01 global budget. Easy degradation is nevertheless
0.099/0.001=99, or 9,900%. This constructed test isolates the global cap; it does
not claim the assignment meets every other policy constraint.

Even perfect prediction of global expected harm cannot in general imply a
conditional relative constraint with another denominator. Empirically, all 72
outcome-defined known-within subgroups in this study still fail the relative
easy check. Using those groups for inference would itself require future labels.

## Missing Labels And Denominators

For unavailable selected-agent ADE, retain an unknown nonnegative contribution.
The sum of observed a_i*h_i divided by original n is a lower bound, not a
zero-filled estimated risk. If it already exceeds the budget, every nonnegative
completion exceeds it. If it does not and any selected cost is missing, the
budget status is indeterminate. If no selected costs are missing, excess on
unselected agents is exactly zero even when their future labels are absent,
because the prediction equals B by construction.

Mean query risk weights queries equally; mean past-agent risk uses total harm
divided by total n. Neither silently replaces the complete-label primary ADE.
All future-defined groups remain descriptive. The unknown-label population is
not exchangeable with the observed-label population by assumption.

## Requirement For A Future Repair

A sufficient, more conservative counterpart to the easy criterion would be

```text
sum_i w_i * a_i * max(d_i,0) <= rho * sum_i w_i * L(B_i,Y_i).
```

For fixed evaluation weights and distribution, an expectation-level counterpart
requires conditional moments such as E[w*h | X] and E[w*L(B,Y) | X], where X
contains only approved past inputs. Multiplying independently estimated easy
probability and unconditional harm is not generally equivalent: easy and harm
can remain conditionally dependent. Learned estimates would still be estimates,
not a finite-sample certificate. Missing-label eligibility, site weights,
denominator uncertainty and the available independent calibration sites all
need an explicit protocol.

This is a mathematical requirement to clarify the next design, not authorization
to fit it on the already inspected outcomes. Independent scene calibration and
confirmation require genuinely unexposed support; renaming existing records or
using overlapping windows does not supply it. The pending primary-metric choice
must be resolved before fixing new loss/selection/calibration targets.

## Claim Boundary

The result demonstrates a failure of the current estimate/constraint combination
on one explored development site. It does not establish which new estimator will
work, a universal impossibility for scene-level selection, a physical safety
guarantee, or a novel theory. No thresholds, training targets, deployed models,
primary metrics, coordinates or time units are changed. No Stage5C or SMC.
