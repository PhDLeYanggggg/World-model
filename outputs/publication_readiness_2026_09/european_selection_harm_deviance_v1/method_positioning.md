# Loss Positioning And Scope

The loss family is classical, not an M3W invention. Banerjee, Guo and Wang,
[On the Optimality of Conditional Expectation as a Bregman Predictor](https://doi.org/10.1109/TIT.2005.850145),
IEEE Transactions on Information Theory51(7):2664-2669,2005, studies conditional
expectation under Bregman loss. The [author-hosted earlier2003draft](https://www.dam.brown.edu/people/huiwang/research/bgw.pdf)
includes the generalized I-divergence example in Table1 and its conditional-mean
result in Theorem1. Both were read on2November2026; the draft and final
publication are distinct versions. No new general theorem is claimed here.

## Local Optimization Hypothesis

For a positive predicted cost p=exp(ell), observed cost y and fixed SOURCE
head scale s, write u=p/s and v=y/s. The two scalar losses are:

```text
square = (u-v)^2
deviance = 2*(u-v+v*log(v/u))
d square / d ell = 2*u*(u-v)
d deviance / d ell = 2*(u-v)
```

At very small u and a positive label, squared-loss logit gradients can become
tiny. The deviance has a different gradient scale. Stable decoder logs allow
this calculation without first taking log of an underflowed floating-point
probability. The derivative calculation is an optimization motivation, not
proof that numerical underflow caused the measured real-data failures.

The repository's historical easy-harm-deviance code already implements stable
logs and its144fit result was negative. This experiment reuses that scalar loss
on the current complete nine-producer cost model, controls easy-only against
dual H/EH replacement, and retains exact original controls. It is a controlled
adaptation, not a newly discovered stable-loss method.

## Why The Classical Property Is Not A Safety Result

The actual network has finite capacity, bounded/coherent coupled cost outputs,
shared reference heads, query weighting, partial labels, regularization and an
additional gain loss. The unconstrained conditional-mean property does not
prove fitted conditional costs are correct, label missingness is ignorable,
or a selected action/recording satisfies2%error-increase risk. Training labels
are observed available-future costs, not identified unobserved world outcomes.

Selection changes the relevant prediction region. Better marginal regression,
a lower fitting objective or finite stable gradients can still lose policy
utility or violate original risk. Therefore the registered paired development
comparison, full recording checks and future independent evidence remain
necessary. No cost-loss theorem supplies a new JEPA/Transformer contribution,
physical safety guarantee, population certificate or world-model success.
