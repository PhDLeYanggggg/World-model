# Forecast Shape at a Fixed Realized Budget

Can a neural model improve the shape of the incumbent forecast without
moving farther from the CV fallback? The preceding same-coefficient
comparison could not answer this: most of its learned edits moved farther,
and the advantage over the incumbent vanished when radii were equalized.

Use the existing exposed OpenDD 8-to-12 development cohort. Keep all
seven leave-location-out folds and seeds 17, 29, 43. The incumbent is the
previous frozen 5% self-history blend. For every fit, a training location's
incumbent path is supplied by the existing producer that excludes both
that training location and the held location. Held paths use the existing
outer producer. Verify registrations, identities, training rosters and
hashes. Future positions and easy membership are supervision only.

For each eligible target, let \(r\) be the incumbent's maximum distance
from CV across the 12 steps. Feed the causal history and the incumbent's
unit-radius offset in heading coordinates. A zero-initialized head learns
bounded velocity increments to that shape. Normalize the new shape to
unit radius inside training and then multiply by \(r\). Compare CV, the
incumbent, and a causal hindcast direction scaled to the same realized
radius. Keep identical targets and realized radius for all edited arms.
The hindcast control may be scaled up; it is a direction control, not
a newly deployed policy.

Train four arms, with the same initialization and 600 site-balanced
updates: observed error, observed plus missing-step disagreement against
the incumbent, the latter plus the fixed positive-easy-harm penalty
against CV, and that same objective with up to two nearest full-history
pedestrians per query. The last two differ only in past neighbor input.
Measure conservative gain vs incumbent, CV and matched hindcast; original
2% flight risk; and the observed relative-vector forecast error over
same-query pedestrian pairs (descriptive, not physical safety).

Primary contrast: self-only easy-harm arm vs incumbent. The peer arm is a
fixed interaction contrast vs the self-only easy-harm arm, not a selection
rule. A gain claim requires the paired 95% location-bootstrap lower
endpoint to exceed zero (3,000 draws, averaging seeds first). The original
recording risk test is separate; undefined views remain failures. No
target-data tuning, external transfer, deployment, Stage5C or SMC.
