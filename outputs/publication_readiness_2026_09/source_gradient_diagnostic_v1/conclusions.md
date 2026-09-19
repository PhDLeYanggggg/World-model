# Frozen training-gradient diagnosis

## Completed Evidence

`fresh_run`: every final importance-corrected checkpoint, all four training
complements, both arms, seeds 17/29/43. Twenty-four full population gradients,
6,144 independent-RNG minibatch probes, fixed output-shrink diagnostics. All
24 diagnostics replay exactly in a second process. No optimizer updates or
new held forecasts. The predecessor checkpoints are `cached_verified`.

| Finding | Result |
| --- | --- |
| Full training-gradient norm | 80.0423 to 117.5362 |
| Fraction of sampled batch gradients above cap 5 | 100% for both proposals |
| Clipped-mean/full-gradient cosine, uniform proposal | 0.998577 to 0.999969 |
| Clipped-mean/full-gradient cosine, corrected episode proposal | 0.999295 to 0.999906 |
| Minimum final-layer squared-gradient share | 99.998390% |
| Static contribution projected onto full gradient | 66.4713% to 80.1524% |
| Static/nonzero contribution cosine | +0.750048 to +0.944888 |
| Training radius/loss-scale median across complements | 527.33 to 626.51 |
| Fixed shrink-to-zero check | Zero output has least loss among tested multipliers in 21/24 heads |

The other three heads, all centered/hyang-complement, have small positive
training gains and prefer .5 or .75 among the fixed multipliers. No multiplier
is selected for held evaluation or deployment. Static harm is reported in
annotation pixels; its percentage over a zero-error baseline is undefined.

## Interpretation

The evidence does **not** support a large reversal of the average gradient by
cap-5 clipping at these final checkpoints. It is not a proof about all earlier
updates, unbiased clipped gradients or Adam dynamics. Finite Monte Carlo means
cannot distinguish every small clipping effect from sampling error.

Static and nonzero-target contributions mostly point in the same parameter
direction here; the simple account that their gradients are in opposition is
not supported at these iterates. The output layer dominates. Combined with the
large restoration/loss-scale ratio, this motivates a single train-scale readout
parameterization comparison. It does not prove that input information is
sufficient or that the proposed repair will improve excluded-scene prediction.

The complete numerical record is [audit.json](audit.json). Per-parameter vectors
and original checkpoints remain private local artifacts. The next comparison
is fixed in [its decision](../source_conditioned_readout_decision.md), not
selected from new held scores.

## Why More Movement Alone Cannot Help

The four training complements have fully static future-label fractions of
61.3775%, 52.5129%, 53.1958% and 55.9112%, respectively. These are loss-audit
statistics, not inference features. For a single unconditional trajectory z
used on every row, triangle inequality gives

`mean_ADE(z, y) - mean_ADE(0, y) >= (2*p_static - 1) * mean_t ||z_t||`.

The coefficient is positive in each complement (0.22755, 0.05026, 0.06392,
0.11822), so unconditional movement cannot improve this uniform training risk.
The existing positive loss-scale divisor does not change that conclusion.
This elementary bound is **not** a new theoretical contribution, and does not
apply to an input-conditioned model as if every row had the same prediction.
It does not prove that the histories are uninformative. It explains why a
successful repair must discover conditional motion evidence instead of merely
increasing average displacement or reweighting future-moving labels.

## Boundaries

Source development only; four already explored sites and shared training folds.
Main, outer, bookstore and external evaluations remain unscored in this step.
Offline supplied annotations, not a strict sensor-as-of certification. Eight
observed / twelve predicted steps, stride twelve raw frames, pixel coordinates;
no seconds, metric, true-3D or foundation claim. No deployment, Stage5C or SMC.
