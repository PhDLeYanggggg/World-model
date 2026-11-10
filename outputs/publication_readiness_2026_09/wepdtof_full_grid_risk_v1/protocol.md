# WEPDTOF: full-grid risk and accuracy sensitivity

Use only the 15 already scored clips from 13 source groups, frozen
past-only forecasts, pilot cutoff and original online label matches.
Keep the four matched controls, per-query edit counts, references and
2% release rule fixed. No training, threshold selection or new cohort.
Recompute the archived all-edited and selected-easy ratios and both
accuracy contrasts before calculating the new quantities.

For full-grid risk use the same all-edited and easy-membership
calculation as [Songdo](../songdo_full_grid_easy_risk_v1/protocol.md),
now for any nonempty edit set. For each accuracy reference, the
full-grid utility contrast uses the original observed CV denominator.
Add or subtract the distance between forecasts at each missing point,
using the same query weights. A strictly negative upper endpoint
against either reference excludes positive utility against both under
every completion. Two positive lower endpoints establish positive
utility against both; this combines with the full-grid risk check only
when the latter passes for every completion. All remaining cases are
unresolved by these sufficient checks, not classified as feasible.
The exact worst-case risk and utility endpoints do not necessarily
share the same completion. No-edit views are neither risk passes nor
strict improvements over CV and are kept on the original roster.

Counts use clip-seed views. The paired differences in support and
pass rates use 3,000 original-source resamples after averaging clips
and seeds with matched edits inside each source. These intervals
describe source sampling, not missing position uncertainty.
