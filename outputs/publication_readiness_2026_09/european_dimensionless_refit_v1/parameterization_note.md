# What the Parameterization Does and Does Not Guarantee

This is an implementation argument, not a new risk-control theorem or evidence
of learned dynamics. Let x denote only observed geometry, with times, masks and
requests fixed. Let B(x) be the selected causal rollout, S(x) the observed
conditioning scale, R(x) the per-request past-motion budget, and
q(z)=z/sqrt(1+||z||^2). The matched models differ only in:

```
control: F_old(x) = B(x) + R(x) q(S(x) f(x/S(x)))
candidate: F_new(x) = B(x) + R(x) q(f(x/S(x)))
```

For positive uniform coordinate rescaling c, suppose the conditioning clamp
is inactive before and after rescaling. Then S(cx)=cS(x); the same selected
baseline and motion budget satisfy B(cx)=cB(x) and R(cx)=cR(x). Conditioned
geometry is unchanged. Therefore F_new(cx)=cF_new(x), in exact arithmetic.
The control generally does not satisfy this equality because q(cz) is not
cq(z). The real-input probe checks finite-precision behavior with the original
tolerances rather than assuming numerical equality from the argument.

This argument does not cover rotating the scene, changing aspect ratio,
changing temporal stride, changing agent detections, or crossing the one-unit
conditioning clamp. It is not a general domain-invariance theorem. Training
also changes because correction amplitudes and gradients change; an accuracy
contrast cannot isolate those mechanisms by itself.

## Bounded Output Is Not Relative Safety

For any future position y, the reverse triangle inequality gives
`abs(||F-y|| - ||B-y||) <= ||F-B|| <= R` for each request. This is an absolute
geometric envelope, not a2% relative-error guarantee. The actual baseline
error is unknown at inference and can be zero. If F differs from B and y=B,
the baseline is perfect and the alternative has positive error. No nontrivial
intervention can promise no harm for every possible target without additional
assumptions on the target distribution.

Consequently, a causal budget can prevent unbounded displacement but cannot
replace reliable gain/harm learning, support-aware abstention and independent
statistical calibration. Silver image trajectories also cannot establish
physical collision safety. The current task measures forecast error, not
actions applied to a physical system. No future label enters these arguments
or the deployment inputs.
