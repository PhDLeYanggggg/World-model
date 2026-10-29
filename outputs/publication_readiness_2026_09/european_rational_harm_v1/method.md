# What the Rational Repair Changes

This is a small expected-cost head over frozen causal features and forecasts,
not a newly trained trajectory world model. The research question is whether
local harm-head gradient obstruction explains unsafe selection. A successful
numerical repair alone would not establish this hypothesis or paper novelty.

Let B,H,R,Re,He denote predicted benefit, positive harm, reference error, easy
reference error and easy positive harm. The source-normalized causal envelope
is E; the protected benefit gives cap=max(E-B,0). The branch learns two raw
logits r and s from a separate causal MLP, leaving B,R,Re bit-exact.

For0<q<1, define the anchor-exact bounded fraction f by:

```
f(q,r) = q + (1-q)r/(1-q+r)     if r >= 0
         q^2/(q-r)              otherwise

df/dr  = (1-q)^2/(1-q+r)^2      if r >= 0
         q^2/(q-r)^2            otherwise
```

Both derivatives equal1 at zero; f(q,0)=q. At finite r, the derivative is
positive in exact arithmetic for interior q. The limits are0 and1. The new
predictions are H'=cap*f(min(H,cap)/cap,r) and He'=H'*f(He/H,s), with explicit
zero-denominator guards. Hence0<=He'<=H', B+H'<=E and B/R/Re unchanged.

The upper-tail expression is evaluated as1-(1-q)^2/(1-q+r) when r>1-q;
this avoids subtraction of almost-equal derivatives in the quotient rule.
The near-zero expression preserves tiny anchors. Float64 ratio arithmetic is
used inside the float32 network, with safe inactive torch.where denominators.
Boundary q=0 or1 necessarily permits one-sided flat behavior; floating-point
underflow and very small gradients remain possible. No claim that all weight
gradients remain informative or that this guarantees optimizer convergence.

The predecessor hard-clips q+tanh(s) and H+cap*tanh(r). On observed fitting
inputs it often reaches exact-zero easy-harm Jacobian. The rational replacement
changes mapping geometry and tail approach together. It is not a unique causal
test of clipping alone; matched initialization, sampling, loss and budget remove
several other differences but not every optimization effect.

Training retains primary five-moment regression, regret and dual-weighted
positive-risk excess. The frozen causal selector requires predicted B>H,
H<=0.02R and He<=0.02Re. Unknown outcomes are retained at evaluation; support
is checked with completion bounds, not discarded to manufacture a safety pass.
The registered gate and all seven controls are unchanged during readout.

Data: exposed original European TRAIN only, obs8/pred12, stride12 raw frames,
detector-track pixel/dataset-local labels. No new independent role, metric,
seconds, physical safety, human gold, true3D, foundation or deployment claim.
No latent generation or SMC. This numerical mapping is not claimed as a novel
standalone method or a substitute for producer-excluded risk calibration.
