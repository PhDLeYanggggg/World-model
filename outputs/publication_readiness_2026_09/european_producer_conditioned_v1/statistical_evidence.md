# Statistical Scope

The primary comparisons use identical candidate and fallback forecasts for the
new global, producer-tagged and placebo-tagged heads. Each head is trained with
seeds17/29/43 under three fixed source folds and two risk events. Each producing
bundle is evaluated separately. No better-half, better-seed or better-event
selection is permitted.

For each locality and subset, gain is 100 times one minus model mean error divided
by reference mean error. The reported aggregate averages supported locality gains,
not overlapping windows. A common locality roster is supplied to each comparison.
The 95% interval uses 3,000 paired source-locality bootstrap resamples, seed39271.
Rows lacking the required future labels do not become zero-error observations.

Every view contains eight excluded development localities. Across views there
are twelve already-opened localities, not 36 independent datasets. Producers,
folds, seeds, rows and locality rosters overlap. Counts of positive or negative
intervals in results.md are descriptive and do not adjust for multiple testing.
These are conditional development intervals, not an independent replication or
a deployment-risk certificate. Seed ranges are retained rather than selecting
one seed to present.

Complete-label ADE and endpoint FDE are reported alongside partial-label ADE.
The locality CSV files preserve tail errors, supported rows and adverse slices.
Unknown-label switch counts remain visible even though those rows cannot support
an accuracy claim. Positive-easy degradation is a net error comparison against CV;
the selected positive-harm moment ratio is a different quantity. Neither replaces
independent risk calibration.

The two-source branch floor, original four-source floor and original stopping
controller have distinct meanings and are never swapped silently. Primary tag
contrasts hold both forecasts fixed. Contrasts with the old stop controller also
include the effect of changing the producing models and their training sources.

The post-readout easy-error decomposition is an explicitly exploratory arithmetic
check on the frozen predictions. It uses the same positive-CV denominator for the
branch floor and the neural increment. It does not refit thresholds, change a
policy, select a model or create a new confirmatory test.

The data are image-pixel, raw-frame EuropeanSquares detector tracks. This study
does not establish metric scale, physical time, physical safety, human-gold
annotation, true3D or foundation-model capability. Independent model-selection,
calibration and confirmation roles remain closed; Stage5C and SMC remain off.
