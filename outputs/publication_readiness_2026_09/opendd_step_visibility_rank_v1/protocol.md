# Predict where forecast edits incur missing-label debit

Use the same fixed joint proposals and 126 OpenDD development directions
as the movement-matched study. For each unordered pair of held and
calibration locations, train on the other five locations. The target is
the binary missing-label mask at each of the 12 future sample steps. No
future positions, future track lengths, or masks enter held inputs. Features
are the original eight-step causal motion summaries and current query size.
Use the same three seeds, 200 fixed updates per fit, four CPU threads, no
workers, and checkpoint every 100 updates. No validation hyperparameter
search. The fixed calibration locations are also excluded from training.

Predict each edited actor's stepwise missing probability. Multiply it by
the step's CV displacement, sum across steps and both actors, and divide
by twelve times the sum of their peak displacements. Subtract this
dimensionless cost from the frozen support-weighted merit. This is an
additional ranking penalty, not an error bound or a new risk estimate.
The profile-only control uses the average training-site missingness at
each step in the identical calculation. Compare both to the existing
movement-matched support-weighted mask. Keep counts exactly matched and
both displacement budgets within 1% of the original merit mask. Do not
alter trajectories or the original three risk checks.

Measure Brier score on the fixed proposed actors, normalized debit MAE,
and direct conservative ADE gain of learned minus profile-only (primary).
Also report relative-motion gain, original-risk and dual-pass counts,
undefined easy risk, and paired gain against support-weighted merit.
Average repeats within seven locations and use the existing 3,000 paired
location draws. Fix scores and masks before reading held labels. This is
an exposed development test, not a new deployment or external test.
