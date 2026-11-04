# Amplitude-Cost Structure And Its Limits

This is elementary task-specific algebra, not a new input-convex-network or
conformal-risk theorem. The registered empirical comparison must still pass.

## Fixed Histories, Paths And Observed Labels

For one agent, let b be the frozen causal physical path and d=p-b the frozen
causal neural displacement. Set p(alpha)=b+alpha*d, alpha in [0,1]. For a fixed
nonempty observed-label index set V, define

L(alpha) = mean_{t in V} ||b_t + alpha*d_t - y_t||_2.

V is a supervision/evaluation mask, not an inference input. Its dependence on
future annotation availability does not change the following within-sample
algebra: the same mask is held fixed as alpha varies. This loss does not
identify the unobserved steps of a partially annotated future.

L is convex because norms of affine functions are convex and averaging
preserves convexity. Positive harm is H(alpha)=max(L(alpha)-L(0),0), so H is
convex, nonnegative and H(0)=0. For 0<a<b<=1, convexity gives

H(a) <= (a/b)*H(b), hence H(a)/a <= H(b)/b and H(a)<=H(b).

Let E=max_t ||d_t||_2, a past-conditioned finite discrepancy bound. The reverse
triangle inequality gives |L(alpha)-L(0)|<=alpha*E. Thus both positive harm
and positive benefit are bounded by alpha*E; their sum is also bounded by it.
Benefit B(alpha)=max(L(0)-L(alpha),0) is not assumed to be a convex harm curve.

R=L(0) is the floor error and does not shrink with alpha. The easy event I is
defined by the unchanged original CV error/cutoff, not by a small-amplitude
candidate. ER=I*R is also amplitude-independent. EH=I*H and NH=(1-I)*H are
both nonnegative convex functions of alpha and sum to H. Conditional expected
H/EH/NH given the same causal history inherit this shape. Their bounded
discrepancy ensures finite harm moments for fixed finite E. None of this
makes a learned predictor accurate or certifies a reference-error estimate.

Wholly unknown label rows instead carry conservative discrepancy contributions
H=EH=alpha*E and no observed R/ER contribution. These are bound-accounting
labels, not fabricated zero future error or identified expected true moments.

## Finite-Grid Hinge Representation

For knots t_j in {0,1/32,1/16,1/8,1/4,1/2}, let

phi_j(alpha)=max(alpha-t_j,0)/(1-t_j).

Each phi is nonnegative, convex and nondecreasing, zero at alpha0, one at
alpha1, and bounded by alpha on [0,1]. For a nonnegative cap c<=E and simplex
weights w_j, the function c*sum_j w_j*phi_j has the same properties.
Splitting its coefficients by q_j in [0,1] produces

EH(alpha)=c*sum_j w_j*q_j*phi_j(alpha),
NH(alpha)=c*sum_j w_j*(1-q_j)*phi_j(alpha).

Both are convex and sum to H. This respects the true easy/non-easy moment
decomposition rather than merely imposing EH<=H at individual grid points.

For a convex nonnegative piecewise-linear curve with value zero at alpha0,
its initial slope and nonnegative subsequent slope increments give the
ordinary nonnegative hinge coefficients. Normalizing by the value at alpha1
gives the simplex form. The same argument applies to EH and NH separately,
then adds their coefficients to construct H. This is a finite-grid structural
representation argument, not a universal approximation claim about the finite
causal encoder or optimization. Softmax/sigmoid parameterizations approximate
coefficient boundary values in their closure; saturation can still impede fit.

In the implemented head, c retains the original full-action softmax harm cap.
Weights and easy fractions are supplied by the existing logits; no parameter
is added. Shared R/ER stay as in the original network. Benefit retains its
original readout but is clipped to R and remaining discrepancy capacity.
The protocol discloses stable hinge arithmetic and FP32 loss reduction.

## Why Shape Is Not Sufficient

- A convex curve can be badly underestimated everywhere. Shape cannot replace
  magnitude learning or policy-conditional reliability.
- A coherent expected cost is not a per-recording realized-risk upper bound.
  Finite samples, rare events and distribution shift still matter.
- The original complete gate requires finite selected easy reference support.
  A record with no observed easy event does not acquire support from a decoder.
- Changing which agents/actions are selected as a control varies does not
  inherit monotonic risk for a fixed selection set.
- Convex positive-harm targets do not imply that the full joint allocation,
  same-count utility, benefit or a learned threshold search is convex.
- Repeated overlapping windows/source contexts are not independent calibration
  observations or verified independent geographic sites.

A label-aware arbitrarily small nonzero alpha may satisfy a positive-reference
row's harm ratio using alpha*E<=0.02*R. This is not a causal learner, a measured
grid result or useful dynamics lift; R and the easy event are future-dependent
labels, the legal grid is fixed, and negligible changes cannot substitute for
the registered utility comparison. At alpha0 the intervention disappears and
selected-support semantics change. Empty support is not a measured safe zero.

## Current Method Claim

The experiment asks whether this structural prior improves learned admission
under a fixed candidate bank and unchanged risk evaluation. It does not train
a new trajectory backbone or execute a generative rollout. The scene-level
solver pools predicted additive budgets, not optimized collision pair costs.
Independent risk calibration and final confirmation remain separate evidence
requirements. No general CRC conditions are established here.

See the [primary-source positioning note](../moving_amplitude_shape_v1/literature_position.md)
for existing convex/monotone neural methods. Even a positive development
contrast would not by itself establish novelty, deployment or submission
readiness. Units remain pixels/native annotation steps, not verified metres,
seconds, true 3D or foundation success. Stage5C execution and SMC stay off.
