# Current-query allocation and the limits of predicted risk

This note defines the registered experiment and its algebra. It does not claim
a novel optimization algorithm, calibrated safety, an independent result or a
completed paper. Numerical results belong to the separately verified readout.

## 1. Task and measurable information

At current raw frame t, Q contains the allowed target queries from one source
recording. It is not necessarily the full roster of visible agents. For each
target i, the fixed bank supplies six physical forecasts and one neural forecast,
plus an unchanged fallback trajectory f_i. Every path and predicted cost uses
the eight observed steps and permitted static/past context only. The task
requests12 future steps at raw stride12, without a verified time/metric conversion.

Let k denote a non-fallback action and let z_ik be binary. At most one action
can be assigned to a target. The predicted action moments are benefit Bhat,
positive harm Hhat, fallback reference Rhat, easy reference REhat and easy
positive harm HEhat. The net predicted gain is ghat=Bhat-Hhat. Unknown future
labels have no role in inference availability, grouping or allocation.

## 2. Fixed decision comparison

The original independent rule chooses the largest positive ghat for each target
among actions with positive references and both pointwise predicted ratios at
most rho=0.02. A target without an admitted action keeps f_i. Let c_Q be the
number of interventions from this rule in Q.

The matched joint problem is:

```text
maximize  sum_i,k z_ik ghat_ik
subject to
  sum_k z_ik <= 1                         for each target i
  sum_i,k z_ik = c_Q
  sum_i,k z_ik (Hhat_ik-rho Rhat_ik) <= 0
  sum_i,k z_ik (HEhat_ik-rho REhat_ik) <= 0
  z_ik = 0 for unavailable/nonpositive-gain/nonpositive-reference actions
  z_ik in {0,1}.
```

The independent assignment is feasible. Therefore an exact solution cannot
decrease the predicted objective at this intervention count. This statement
concerns model predictions, not realized error. The free-count secondary drops
only the cardinality equality. The uniform control uses one common candidate
for every target when all constraints hold, otherwise every target keeps its floor.

The implementation first checks whether the unconstrained optimum is feasible;
otherwise it solves the binary linear program. It postchecks the raw signed
budgets and assignments. A time-limited, nonoptimal or numerically invalid
solution returns the original independent assignment. Such a fallback is a
valid preserved control, not an exact-optimality result.

Equal counts do not imply equal selected reference mass. Both methods use the
same2% ratio criterion, not necessarily the same absolute harm allowance.
Predicted references can also vary across actions even though the true fallback
reference for a target is common. Reference inflation and harm underprediction
must not be confused with a useful joint allocation effect.

## 3. What composes from query to recording

If the queries partition the evaluated rows and every Q satisfies the two
predicted signed inequalities, summing those inequalities gives their predicted
recording-level counterparts. This is elementary addition of finite moments.
It is not a new safety theorem or a statement about observed outcomes.

There are two additional gaps. First, a zero easy-reference denominator leaves
the selected easy ratio undefined, even if its signed inequality is zero.
Second, marginal cost fit need not remain calibrated after selecting actions
using the full query. A head conditioned on self features need not estimate
the conditional moments given all information used by the allocator.

Even under the stronger hypothetical assumption that the predicted moments
equal the true conditional moments given all current-query information, signed
constraints imply a ratio-of-expectations bound when the expected denominator
is positive. They do not, without additional assumptions and uncertainty
control, imply an expected ratio, a high-probability guarantee or compliance
in every finite recording. None of these assumptions is established here.

## 4. Missing outcomes and the existing certificate

The evaluation holds known partial-label ADE costs fixed. Let H and R be the
selected known positive harm and reference masses, HE and RE their selected
known easy versions, and U the sum of causal disagreement envelopes on selected
wholly unknown rows. The existing upper summaries are:

```text
all_upper  = (H+U)/R    if R>0,  otherwise undefined
easy_upper = (HE+U)/RE  if RE>0, otherwise undefined.
```

For positive R and RE, the two reported upper summaries are at most rho if
and only if `U <= min(rho R-H, rho RE-HE)`. This is an arithmetic factorization
of this fixed certificate, not a claim that its envelope relaxation is tight
over physically realizable futures. It does not bound missing interior points
in partially observed ADE rows or establish missing-at-random labels.

Known-only compliance can fail to certify the complete policy because U is
too large, or because selected support is absent. If known positive harm
already exceeds its budget, reducing U alone cannot repair that channel.
The separate full easy-degradation upper condition and utility requirement
must also be checked. Benefits cannot cancel positive harm in the risk ratio.

All these masses use evaluation labels; they are diagnostic quantities, not
features or a deployable gate. The current allocator cannot use the eventual
known mask, future endpoint or true oracle assignment to satisfy them.

## 5. Evidence and contribution boundary

This experiment asks whether fixed cost estimates become more useful under
joint allocation, not whether another network architecture can be trained.
It also checks a label-aware record witness at the current-frame level, which
is opportunity evidence rather than causal learnability or a matched-rate optimum.

[JFP](https://proceedings.mlr.press/v205/luo23a.html) already models dependencies
between multi-agent future outputs. Our unary budget coupling does not model
pairwise dynamics or reproduce JFP. Generic routing is also covered by
[regression with multi-expert deferral](https://proceedings.mlr.press/v235/mao24d.html).
Any future contribution must be the demonstrated gain/harm decision mechanism,
its treatment of support and uncertainty, and independent matched evidence,
not the existence of a Transformer, a solver or a fallback.

The three seeds and locality bootstrap describe exposed fitting variation.
The independent calibration/confirmation roles remain closed. Only a complete,
compliant later evaluation could support generalization. No deployment promotion,
Stage5C execution, SMC, true3D, foundation, metric or seconds-level claim follows.
