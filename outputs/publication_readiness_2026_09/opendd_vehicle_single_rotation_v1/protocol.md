# OpenDD vehicles: allow one angle change or keep the original

The two-rotation study finds that 1,393 of 1,431 hindsight
interaction wins can be realized by changing the angle of just
one vehicle. Retain the two original edits, recipients and all
per-step displacement norms, but change at most one direction.

For each candidate predict individual and interaction utility
against three past-only surrogates: OLS4, bounded decaying
acceleration, and bounded decaying turn. Use the worst gain
over the surrogates. Relative errors involve edges incident to
the changed vehicle and are weighted by inverse observed distance.
The individual arm acts only on a positive individual score.
The interaction arm requires both scores to be positive, then
maximizes their sum. The matched individual control picks the
best individual angle for exactly the same active vehicle and view.
Otherwise keep both original paths. No outcome-based thresholds.

Evaluate original 2% risk and CV/OLS4 utility, direct conservative
trajectory and relative-motion contrasts on the same 8,874 views,
and the realized rotation rate. Location bootstrap resamples seven
location means after averaging calibration directions and seeds.
No new external labels, training or release.
