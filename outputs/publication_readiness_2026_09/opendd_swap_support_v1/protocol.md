# OpenDD: separate observation support from observed swap effects

Keep the exposed OpenDD seven-location, three-seed roles, native 8/12
steps, and all 8,814 legal frozen nominations of the
[value audit](../opendd_realized_swap_value_v1/README.md). A donor's legal
candidate bank comes from a producer that excludes that donor and the
outer held location. Train on all donor candidates, not on candidates
chosen using a head trained on their outcomes. Use the same six donors
with equal sampling, 300 fixed updates, and 50-step checkpoints.

For individual paths and affected relative pairs separately, learn the
fraction of the movement bound on observed steps and the conditional
positive and negative error difference on that support. Multiply the
conditional signed effect by support and debit the unobserved share to
recover the direct lower endpoint. Zero support has no conditional loss.
Predict the sign of the lower endpoint as a separate binary target. Use
two otherwise identical heads: existing features with nine zero columns,
or the past-only footprint proxy in those columns. The footprint is the
observed rectangle of histories in a shared query frame, not a true
camera boundary. No actual future support enters inference.

Primary comparison: footprint sign versus the v196 fused score for
relative positive-gain ranking. Report AUC, precision-recall, support
error, and the geometry ablation, and compare individual ranking to the
archived v194 gain. Aggregate repeated directions within site and seed,
then resample seven sites 3,000 times. The predeclared screen for a
further allocation experiment is a positive relative AUC interval
against v196 and no drop in individual AUC point versus v194. This
experiment itself makes no new allocation or release decisions. The
unchanged 2% recording rule is not reestimated or weakened. No test
threshold or independently reserved outcome is used.
