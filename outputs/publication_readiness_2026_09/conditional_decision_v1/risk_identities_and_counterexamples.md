# Conditional Risk Identities and Executable Counterexamples

2026-09-21. Mathematical clarification only. The finite examples are synthetic,
not SDD/ETH/UCY measurements, a trained method or an independent safety certificate.
They change no registered metric, risk tolerance, model, policy or data role.
No novelty is claimed for conditional expectation, Cauchy-Schwarz or these
constructions. Related prior work and the scope of inspected proofs are recorded
in [the source review](prior_work_and_method_boundary.md).

## 1. Why Global MSE Can Miss the Action Region

Let h >= 0 be realized positive harm, h_hat(X) its estimate and a(X) in {0,1}
a fixed deployable intervention rule. X includes all information used by that
rule; future labels do not enter a. For a fixed population with finite second
moments and coverage c=E[a]>0:

```text
E[a*h] - E[a*h_hat] = E[a*(h-h_hat)]
|E[a*(h-h_hat)]| <= sqrt(c * E[(h-h_hat)^2])
|E[h-h_hat | a=1]| <= RMSE / sqrt(c).
```

The first bound follows from Cauchy-Schwarz and a^2=a; divide by c for the second.
Thus an absolute error bound over the whole population can be uninformative
inside a small intervention group. Small *relative* MSE compared with a poor
constant is even less of a guarantee. These are population identities, not
estimates with finite-sample confidence, and they say nothing about arbitrary
distribution shift. c=0 gives analytic baseline fallback, not positive gain.

**Exact witness 1.** Three observable strata have probabilities .01, .49 and
.50. Their true harm is 1, 0 and 10; predicted harm is 0, 1/49 and 10. Global
true and predicted mean harm both equal 5.01. Harm MSE is 1/98, versus 24.9099
for the true-mean constant. Predicted benefits are .1, 0 and 0, so the fixed
gain>=.02/harm<=.05 gate selects only the first stratum. Its predicted harm is
zero, actual harm is one and actual net gain is minus one. Realized benefit is
zero in every stratum; these are consistent positive-part loss targets.

Correcting the *global* mean cannot fix this example: it is already correct.
The example does not show that any particular fitted M3W architecture cannot
learn the missing distinction. It only refutes the proposed implication from
low aggregate MSE or a correct mean to reliable switching.

## 2. The Conditioning Information Must Cover the Policy

If mu(X)=E[h|X] exactly and a is X-measurable, then E[a*h]=E[a*mu(X)] by iterated
expectation. Under finite second moments, the unrestricted population MSE
minimizer is precisely mu. Accordingly, the diagnosis does **not** show that
squared regression is mathematically unsuitable. It concerns finite model fit,
available information, target weighting and the unsupported use of global fit
as a decision-reliability certificate.

Calibrating only a score q means E[h|q]=q. For a policy depending on additional
scene information Z, generally E[h|q,Z] != q. A joint selector uses other
agents, membership, geometry and competing actions. Calibrating an isolated
agent score therefore need not calibrate the composed action. One can instead
study the full-scene conditional cost or a prespecified class of whole-policy
weighted residual moments, but finite-sample certification still needs support.

**Exact witness 2.** Four equally likely states have h=(1,1,0,0) and the same
score q=.5. The only score bin is perfectly calibrated. A past-observable Z
equals (1,1,0,0), and a=Z. Predicted population selected harm is .25, actual
selected harm is .5. The causal feature is deliberately predictive in this
synthetic law; the policy does not inspect an outcome. The problem is discarded
context, not leakage. With the true E[h|Z], the identity is restored.

## 3. Easy Preservation Is a Joint Outcome Moment

Let L_B and L_N denote nonnegative losses of fixed baseline and candidate;
d=L_N-L_B, h=max(d,0), and w=1{L_B<=tau}. For binary intervention and a separable
trajectory loss, realized policy loss is L_B+a*d. This equality need not hold
for a joint collision/interaction loss, which must be scored separately.

For a *specified population aggregation* with D=E[w*L_B]>0, relative easy
degradation <=rho is exactly

```text
E[w*a*d] <= rho * E[w*L_B].
```

A conservative sufficient condition replaces d by h. The outcome-defined w
is a training/evaluation label, not an available inference gate. A past-only
model could predict joint conditional moments

```text
q(X) = E[w*h | X]       r(X) = E[w*L_B | X]
E[a*q(X)] <= rho*E[r(X)]  =>  relative easy degradation <= rho,
```

provided these are the true moments for the specified population and aggregation.
For estimated moments, both the numerator's upper error and denominator's lower
error matter. For example, independently justified bounds
H<=H_hat+epsilon_H and D>=D_hat-epsilon_D>0 suffice when
H_hat+epsilon_H <= rho*(D_hat-epsilon_D). No such empirical bounds have been
established here. This algebra is not an algorithm or newly proved calibration
theorem. Any new target requires the pending metric decision and registration.

In general,

```text
E[w*h|X] = E[w|X]*E[h|X] + Cov(w,h|X).
```

**Exact witness 3.** At one observed X, two equiprobable outcomes have baseline
losses (.01,1), harms (.01,0) and easy indicators (1,0). The correct joint moment
is .005, but the product of the two correct marginal means is .0025. Perfect
separate easy and harm predictions can still understate the required joint
quantity. The illustrative 2% easy budget is .0001. This is not evidence that
the covariance has that value in the real data.

## 4. Aggregation Is Part of the Scientific Target

For multiple scenes, E[scene numerator]/E[scene denominator] is not generally
E[scene numerator/scene denominator]. A scene-wise sufficient condition implies
the analogous bound for every positive-denominator scene, but an aggregate
condition does not. Within-scene agent weights, across-scene weights, masks and
missing labels must all be explicitly fixed. Outcome-dependent denominators
cannot silently become inference features.

**Exact witness 4.** Two equally weighted one-agent scenes have baseline losses
(.01,1) and policy excess losses (.01,0); both are easy for the toy threshold.
The pooled relative degradation is 1/101, about .9901%, while the equal-scene
mean of relative degradations is 50%. The former passes 2%, the latter fails.
This example selects neither as the project's primary estimand. It demonstrates
why the pending aggregation decision is consequential, not cosmetic.

If the specified baseline denominator is zero, relative degradation is undefined.
Neither an epsilon nor exclusion is silently allowed. A conditional-moment head
and even a correct pooled bound cannot be advertised as a different scene-level
or worst-case guarantee. Missing selected labels remain unknown, not zero harm.

## 5. What Was Executed

The verifier uses exact rational arithmetic, without reading caches, source
trajectories, checkpoints or evaluation labels. It reproduces all four witnesses
and exhaustively checks eight binary policies for the finite Cauchy-Schwarz and
positive-harm inequalities, plus four rules for the conditional-mean identity.
It tests the zero-denominator refusal explicitly. These are implementation
checks on finite constructions; the derivations above supply the general algebra.

```sh
.venv-pytorch/bin/python scripts/verify_m3w_conditional_risk_identities.py \
  --output outputs/publication_readiness_2026_09/conditional_decision_v1/verification.json
.venv-pytorch/bin/python -m pytest tests/test_m3w_conditional_risk_identities.py
```

The output includes exact fractions, decimal renderings and the verifier's source
hash. [Execution notes](execution_notes.md) distinguish these tests from the
earlier real-data diagnosis. No new model, threshold search, calibration or test
readout is executed. Stage5C and SMC remain disabled.
