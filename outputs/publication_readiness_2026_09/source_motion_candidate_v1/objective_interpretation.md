# Why Zero Predictions Can Be Correct Under ADE

## Material Passport

Analytic interpretation written during the fixed loss-control run, before its
comparison results. This does not change the registration or claim a new theorem.
No empirical conditional probability is estimated here.

Let X denote all admitted observations and Y the twelve-step displacement
trajectory. Write N(a) = mean_t ||a_t||_2. This is a norm on a fixed-length
trajectory. For a given X, let q = P(Y != 0 | X), where zero means the entire
target path is zero. The conditional ADE risk is

```text
R(a | X) = (1-q) N(a) + q E[N(a-Y) | X, Y != 0].
R(0 | X) = q E[N(Y) | X, Y != 0].
```

The triangle inequality implies

```text
R(a | X) - R(0 | X) >= (1-2q) N(a).
```

Therefore if q <= 1/2, predicting no movement is an ADE-risk minimizer.
For q < 1/2 it is uniquely optimal among distinct deterministic paths.
The converse does not hold: opposing possible motion directions can leave zero
optimal even when movement is common. This is an elementary conditional-risk
observation, not a new M3W contribution or a guarantee about a fitted network.

The marginal fraction of nonzero source targets does not establish q for any
particular observed context. Overlapping windows do not give independent
estimates of these conditional probabilities. We cannot infer that zero is
optimal everywhere, that the inputs are sufficient, or that the task is
unlearnable from a class count or this inequality.

"Nonzero" means a change in supplied annotation coordinates. It is not a
human-verified movement, start-intention or physical event label. Box jitter,
annotation discretization and interpolation can contribute; this experiment
does not estimate how much each contributes or remove them by an outcome filter.

## What the Registered Intervention Tests

Removing zero-target gradients makes the optimized population objective
q E[N(a-Y) | X, Y != 0] instead of R(a | X). The exact objective still uses the
full batch denominator and the same training-only cost scale. It can encourage
a stronger conditional-motion candidate, but intentionally stops charging that
candidate for movement on zero targets during training. The full sampler,
nonzero-row exposure and evaluation cohort remain unchanged.

This may increase oracle candidate utility while worsening actual ungated risk.
Such a result would motivate causal gain/harm learning, not deployment or a
claim that ADE itself is broken. Training-label weighting is not inference
access to Y. All predictions still receive past-only inputs; the actual model
does not know whether the future will move.

Even a large future-oracle gain would require a usable causal selector, nested
producer-safe validation and independent risk evidence. A low oracle gain after
this intervention would instead weaken the specific zero-gradient-suppression
explanation and motivate direction/path or observation-support diagnosis.
Neither outcome proves strict online causality, metric geometry, physical safety
or a world-model contribution. No Stage5C execution or SMC.
