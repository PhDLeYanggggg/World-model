# Does a peer's recent motion improve the fixed-budget forecast menu?

On the existing seven OpenDD development locations, use eight observed
steps and predict twelve steps. Keep the original shared-menu additive
reference, per-query edit counts and peak displacement slots. Keep all
one-edit assignments fixed. Preserve CV as an available action for at
least one recipient. Use the previously scored three seed budgets and
held-source self-history forecast, without retraining them.

For an agent with a nearby co-directional peer in the same observed query,
the new trajectory starts with the agent's eight-step OLS velocity.
The transient is the difference between one-step velocity and the
eight-step estimate, decaying with a four-sample half-life. Compare using
the agent's own transient to using its peer's transient. A peer is the
nearest other agent with heading cosine at least 0.8 and final separation
at most their mean eight-step displacement. No added option for either
arm when that support is absent. Shorten both paths toward CV to the
smaller of their two natural peak displacements. This makes the added
row-slot availability exactly the same between both arms. Neither arm
enlarges an edit to fill a slot.

Freeze both forecast banks from observed histories before scoring. First
measure the exact constrained pair-error frontier for each expanded
menu, using the old expanded menu as the common reference and the same
three 2% checks, two baseline accuracy requirements and no decrease
in conservative individual accuracy against additive. Report full-grid
and co-observed scores as in the parent. Do not apply old menu-specific
infeasibility certificates where a new candidate is eligible.

Then evaluate a label-blind shape replacement that substitutes the extra
path at the same additive recipients and radii when eligible on two-edit
queries. Both arms must substitute exactly the same rows. Compare peer
to own transient using conservative individual accuracy and pair error;
report the original risks versus the unchanged CV reference. Aggregate
at recording-seed, then location, and resample locations 3,000 times for
descriptive paired intervals. No threshold, model, or arm is selected
using these results. A higher hindsight ceiling alone does not establish
an advantage from the causal rule.

No held confirmation sources, new split, risk-budget change, calibration,
training or deployment. This is a test of incremental candidate supply
and label-blind shape substitution on already exposed development data.
