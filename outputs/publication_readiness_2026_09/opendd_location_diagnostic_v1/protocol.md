# Registered OpenDD Location-Held Diagnostic

Question: with better future-label support, can a learned intervention rule
identify where a different causal motion forecast improves on OLS4? Does
local neighbor history help relative to an equal-capacity self-only model?

This is an exploratory forecast diagnostic on published OpenDD v3 position
tracks, under the delegated independent-source research scope. It is not a
new official benchmark, DroneCrowd confirmation, or a policy safety claim.
All seven sites are used in seven leave-one-site-out folds, with no validation
search, test tuning, pooled feature normalization or chosen winning fold.
No site or flight ID is a model feature. A site is the comparison block;
nearby sites are not assumed to be independent cities.

Use pedestrian positions only from the seven hash-bound v3 archives. Use
8 observed and 12 predicted samples at stride 12 on the audited OpenDD export
grid, with no retiming of SDD or DroneCrowd. Within each flight choose the
earliest available query start with at least two complete pasts, skip its
whole 20-sample interval and repeat. Neither future availability nor future
motion chooses the query, recipients or intervention count. Missing future
steps stay missing. Report source positions as published UTM units; neither
raw-video tracking causality nor cross-dataset calibration is asserted.

Use CV (last displacement) and anchored OLS4. Alternatives are OLS3, OLS8,
fixed damped OLS4 and fixed bounded acceleration, exactly the registered
past-motion menu. The primary comparison is learned gain/harm selection vs
OLS4 at exactly one nonidentity edit per supported query. Compare against
minimum forecast change and history hindcast at the same count. Menu oracle
uses outcomes and is diagnostic only. All arms use the same alternative
trajectories, except the fixed CV and OLS4 no-edit references.

Fit a small pointwise Torch model on the other six sites. Seeds 17, 29, 43,
300 AdamW updates, lr 0.001, batch 128; each draw selects a site then a target
uniformly. Two matched arms, self-only and nearest-neighbor past geometry;
identical draw sequences, initialization, parameter counts and supervision.
Targets are normalized paired observed ADE gain and its positive harm part,
computed on available labels and not used as inputs. Rows with no outcomes
do not contribute to the fit. Rank by predicted signed gain minus harm.
Checkpoint every 100 updates; no result-based early stopping. Fixed
comparison: neighbor vs self on the same query/edit count.

Report ADE and endpoint FDE on observed labels. Primary statistic: the
paired gain lower bound vs OLS4 over all 12 scheduled steps, charging
prediction disagreement on every missing step. Report gains in relative
points against observed OLS4 error mass, averaged equally over the seven
sites after averaging seeds, with 3,000 paired site resamples. For neighbor
vs self, bound missing steps using those two actual predictions, not by
subtracting unrelated marginal lower bounds. Report complete-label subset,
no/partial label coverage, switches, all sites and loss traces. Easy harm is
secondary: signed ADE change on positive CV error below the 25th percentile
from complete-label training rows only, with missing or zero support reported.
This descriptive diagnostic does not alter the original per-view 2% safety
test. No stage5C, SMC or deployment promotion.
