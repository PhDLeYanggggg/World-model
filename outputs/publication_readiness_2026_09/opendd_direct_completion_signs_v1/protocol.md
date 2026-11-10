# Two-sided completion for the fixed OpenDD contrasts

Use the frozen observed and lower contributions of the same three
contrasts: support-weighted versus original merit at equal count, at
equal count and movement, and learned missing-step debit versus the
donor step profile at equal count and movement. Use the existing
seven equally weighted location means, not individual windows. No
new score, training, threshold, raw-label access or cohort.

With nonnegative query weights and the same fixed observed CV
denominator, lower = observed - missing-step distance. Compute
upper = observed + missing-step distance on the same direct pair.
Do not subtract two different policies' precomputed lower bounds.
Report the sample envelope and the original 3,000 paired location
resamples of each endpoint separately. Classify signs from the
sample envelope; location bootstrap intervals answer a different
question. Drop one location at a time to test whether a single
location determines whether the sample envelope crosses zero.

The endpoints are attainable if missing positions may vary freely
by scored actor-step: put each missing point at the treatment or
control forecast. This gives equality in the reverse triangle
inequality. The endpoints do not assert physically coherent
trajectories or any model of why labels are missing. Use only the
additive trajectory comparison for this argument, not relative
motion or the nonlinear recording risk ratios.
