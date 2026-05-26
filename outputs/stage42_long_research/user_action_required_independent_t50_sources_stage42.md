# User Action Required: Independent T50 Source Diversity

Stage42-CB found protected t50 gains are source-concentrated. Stage42-CC scanned local files and generated the following required actions.

## CRITICAL

- action: Provide or legally enable at least one independent t50-capable non-SDD top-down pedestrian source.
- why: Stage42-CB found t50 gains are source-concentrated; Stage42-CC found no unused ready-to-claim source diversity repair without split rebuild and license/no-leakage checks.
- official_targets: `['UCY crowd original official source', 'ETH/BIWI original pedestrian sources', 'TrajNet++ official challenge/data access', 'OpenTraj-supported original dataset paths with underlying dataset terms verified']`

## MEDIUM

- action: Use alternate current-source representations only for format repair or split rebuild diagnostics, not as independent new sources.
- why: Same parent/source directory as current data means these are not independent held-out sources.

## HIGH

- action: For any candidate source, rerun conversion, no-leakage audit, train/internal-val policy selection, and final test once.
- why: Inventory is not conversion, and conversion is not benchmark success.

Do not count registry-only, unlicensed, or merely parsed files as converted datasets. Do not claim broad source-level generalization until conversion/no-leakage/source-CV/final test are complete.
