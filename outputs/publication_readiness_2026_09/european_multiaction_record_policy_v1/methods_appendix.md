# Baseline-Relative Multi-Action Decisions: Methods Appendix

Working English manuscript material, 29 October 2026. This appendix specifies
the current experiment; it is not a complete anonymous CVPR submission and
does not claim a successful learned policy before its readout.

## 1. Forecast Substitution, Not Physical Control

For observed agent history and local neighbors x, a frozen causal floor b(x)
and K frozen forecast candidates p_k(x) are available. A decision chooses
a(x) in {0,...,K}, with a=0 retaining b. The word intervention denotes changing
the forecast presented to a downstream user, not commanding the real agent.
All trajectories here are in image pixels on an 8-observed/12-requested grid.
No meter, effective-second or physical-safety interpretation is assumed.

Let c(p,y;M) be ADE on the observed nonempty future label mask M. For each
candidate, define the five supervised moments

```
B_k = max(c(b)-c(p_k), 0)
H_k = max(c(p_k)-c(b), 0)
R   = c(b)
ER  = 1{0 < c(CV) <= tau_train} R
EH_k= 1{0 < c(CV) <= tau_train} H_k.
```

The easy threshold tau_train is frozen from producer-fitting sources. It is
not re-estimated on policy validation or test. Unknown rows have no realized
cost moment label. Their future masks, endpoints and eventual easy membership
are never decision inputs. Predicted rollout endpoints are legal causal inputs;
ground-truth future endpoints are not.

## 2. Continuous Conditional Cost Model

The preceding matched experiment uses a shared MLP on each action's existing
380-dimensional observed geometry/rollout vector and a seven-way action code.
Its outputs estimate B,H,R,ER,EH, rather than a one-hot best-action label.
The causal maximum rollout disagreement bounds predicted B+H. Benefit is
also bounded by predicted R; easy-reference and easy-harm are gated fractions
of their corresponding reference and harm outputs.

All three action-bank arms use identical model dimensions, initialization,
source-balanced row draws and 2,000 updates. Normalization and initial moment
scales use physical fitting actions only. Underestimated harm receives fixed
4x weight in moment regression, with an additional continuous relative-cost
contrast loss. A predicted pointwise 2% gate is only a decision heuristic,
not a theorem about realized selected risk.

The frozen negative result matters: the physical+neural arm gives 0.553787%
known-label ADE gain over the floor on exposed inner-held records, compared
with 0.595285% for physical-only. The paired difference is -0.041498pp,
conditional 95% locality interval [-0.124505,+0.025491]. Only20/186 repeated
recording/head views satisfy full risk conditions. An attractive label-aware
oracle does not establish causal learnability or deployment safety.

## 3. Elementary Completion and Support Facts

For any single time step, reverse triangle inequality gives

```
abs(||p_t-y_t|| - ||b_t-y_t||) <= ||p_t-b_t||.
```

Averaging any nonempty future mask therefore bounds the absolute ADE
difference by D_k=max_t||p_kt-b_t||. The implementation uses the maximum of
this disagreement and disagreement to CV, preserving the prior contract.
For unknown selected outcomes, -D_k is a lower utility contribution and D_k
is an upper positive-harm contribution. These are conservative completion
bounds, not an assumption that unknown cost is zero or a guarantee about
unseen population risk. Partial known masks retain their realized costs.

For a fixed decision, sum known benefit, harm and reference over its selected
rows. Adding unknown selected disagreement to the harm numerator and retaining
only the known positive reference denominator yields a valid upper ratio when
that denominator is positive. The analogous easy ratio treats the unknown
easy event conservatively. If its known denominator is zero, the ratio is
not_estimable; it cannot be reported as zero harm or a passed safety gate.

There is also a simple necessary support condition. If every real eligible
known easy action in a recording has H_k/ER>0.02, any nonempty mixture of those
actions has a reference-weighted average exceeding0.02. Adding nonnegative
unknown completion mass cannot repair the upper bound. If there are no such
known easy actions at all, the selected easy ratio is undefined. This explains
why some fixed binary-neural recording controls cannot pass even for an
oracle, whereas a larger physical action bank can have feasible alternatives.
These observations support protocol checks; they are not claimed as novel
consistency, conformal or generalization theorems.

## 4. Exact-Record Decision Repair

The current bounded repair freezes both trajectory and conditional-cost
models. A compact decision head receives five predicted moments, six existing
motion/neighbor descriptors and a causal envelope, plus action identity. It
is initialized at the preceding pointwise choice. The floor logit is fixed
at zero; exact copies of the floor are unavailable as artificial interventions.

For each fitting recording r, define complete finite-population totals using
the causal hard action, known supervised costs and unknown completion terms:

```
U_r = sum(selected known (B-H)) - sum(selected unknown D)
C_r = sum(selected known (H-.02R)) + sum(selected unknown D)
E_r = sum(selected known (EH-.02ER)) + sum(selected unknown D).
```

Train to increase U_r while enforcing C_r<=0 and E_r<=0 with separate recording
duals. A third fitting constraint requires positive real easy-reference mass,
equal to one minimum supported eligible easy-row reference. It prevents an
empty ratio from becoming a convenient zero-risk training solution. It does
not change the evaluation tolerance or permit an identical-floor pseudo-action.

Categorical straight-through gradients choose one action or floor. The complete
fitting populations, including unknown outcomes, contribute to exact record
gradients and dual updates; chunking changes memory use, not the population
objective. This avoids sampled risk estimates missing rare easy rows. It is
still empirical recording-aware training, not independent risk calibration.
Its 54 final endpoints must be frozen before readout, with no best-seed or
post-readout threshold selection. Results are reported separately.

## 5. Statistical and Prior-Work Boundaries

Primary utility comparisons pair predictions on the same rows, with causal
disagreement bounds for unknown outcomes. Within each locality, average
dependent context/head-seed views before3,000 locality bootstrap draws. Report
the resulting intervals as conditional SOURCE-development uncertainty. Do not
treat overlapping windows, repeated recordings or three cost-head seeds as
independent data or three newly trained forecasters. Count-matched controls
are utility comparisons; only full policies face the complete2%risk gate.

Multi-expert regression and fixed-predictor deferral are established settings,
including [Mao et al., ICML2024](https://proceedings.mlr.press/v235/mao24d.html).
[Cortes et al., ICML2026](https://proceedings.mlr.press/v306/cortes26b.html)
also address expert imbalance with cost-sensitive deferral. Neither a learned
cost bank nor a fallback option is sufficient novelty by itself. Their
published methods remain required matched controls, not methods reproduced
by this MLP implementation.

[Conformal Risk Control, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf)
requires specified exchangeability, boundedness and monotonicity conditions.
Selected harm/reference ratios need not improve monotonically when a cutoff
removes a safe reference row. Its expected-risk guarantee cannot simply be
attached to the present per-record ratios or to previously exposed records.

Scene-joint interaction optimization, published deferral controls, independent
recording calibration/confirmation and full anonymous reproducibility remain
necessary before a main scientific contribution or submission-ready claim.
Stage5C execution and SMC remain off.
