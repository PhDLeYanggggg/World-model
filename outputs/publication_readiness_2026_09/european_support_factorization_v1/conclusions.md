# What The Support Filter Was Removing

## Result

I completed the registered 936-view comparison. The support-filter repair is not supported: none of the four factor guards has a positive all-ADE confidence interval against its unchanged stop controller in either target family. This is a fresh fixed-rule evaluation on cached_verified forecasts, learned heads and data, not new neural training. Deployment remains unchanged.

| Guard | CV-target all gain vs stop | Negative / positive CI | Floor-target all gain vs stop | Negative / positive CI |
|---|---:|---:|---:|---:|
| History | -0.10475% to -0.01530% | 30 / 0 | -0.10413% to +0.00587% | 28 / 0 |
| Disagreement | -0.06695% to +0.00551% | 27 / 0 | -0.06418% to +0.00543% | 26 / 0 |
| Separate marginal support | -0.12444% to -0.01791% | 31 / 0 | -0.11902% to -0.01490% | 29 / 0 |
| Same-source joint support | -0.12791% to -0.01813% | 36 / 0 | -0.12255% to -0.01490% | 33 / 0 |

Each entry spans 36 correlated development views, not 36 independent tests. The small positive points in three cells have intervals including zero. Hard-subset comparisons against stop also have no positive interval for any factor guard. All arms, controls, complete-window metrics, endpoint metrics, locality errors and tails remain in [results.md](results.md) and the JSON/CSV files.

## What This Explains

**Historical rarity is not a reason to reject a forecast by itself.** The history-only rejection category removes more benefit than harm in all 36 views for each controller family. Its loss is 0.00656-0.07514 percentage points of gain over the floor for CV targets, and 0.00540-0.07534 points for floor targets. This category alone rules out blaming every failure on model-generated disagreement.

**Disagreement rejection also removes useful interventions.** Disagreement-only removals have negative net value in 36/36 CV-target views and 34/36 floor-target views; the other two have no such removal cost. Requiring the same sources to support both axes adds a further net loss in 36/36 and 34/36 views. The rejected cost categories exactly reconstruct joint removals.

**Removing a bad filter is not a new model improvement.** History-only and disagreement-only guards beat the joint guard in every point estimate. Disagreement-only gains over joint are +0.00933% to +0.07977% for CV targets and +0.00893% to +0.08032% for floor targets. Yet disagreement-only loses against the unchanged stop controller in 35/36 points for both families. The reference matters.

**Equal-coverage controls do not establish reliable ordering.** Relative to same-frame random controls, all-ADE positive/negative interval counts are history 5/4 and 4/4, disagreement 2/3 and 2/4, separate 4/4 and 4/4, and joint 5/5 and 4/4 for CV/floor parents. Some easy-subset comparisons favor filtering; they are not suppressed, but they do not establish a robust all-policy allocation advantage. Flexible query counts and degenerate queries are published separately.

## What Remains True

The unchanged stop controller still improves all ADE over its actual fallback by 0.12488-1.76335% for CV targets and 0.12253-1.64105% for floor targets, with positive conditional intervals in all 36 views per parent. These gains preceded this experiment. They cannot be credited to support factorization.

The latest-step stopping repair remains intact in every policy. No supported zero-CV case is harmed, but there are only four unique such rows from one locality, two future labels each and no endpoint. Across all 936 views the worst positive-easy locality degradation is 0.43686%, below the empirical 2% limit. The floor-target stop controller still has a worst hard-locality loss of 2.80551% against its floor. Neither result is a physical-safety certificate.

## Verification And Next Decision

936 causal decisions, 288 old decision anchors, 144 independent coordinate arrays, 12,528 independent metric reductions, 1,728 old metric matches and 4,608 partition equalities pass. All 326 tests in the explicit 53-file scope pass; the unrelated legacy suite was not run. Both decision banks completed before the first new readout. The evaluation phases took 664 and 693 seconds on native arm64 CPU, without resource failure or downscaling.

I will not promote percentile support rejection or search new cutoffs on these outcomes. The next targeted experiment should separate producer transport from utility learning using source-excluded, producer-matched forecasts and controls. It must measure whether matching the fitting and inference producers improves gain/harm ordering, with fixed stopping protection and unchanged evaluation roles. The current decomposition motivates that test but does not establish its answer.

These are opened-development, image-pixel obs8/pred12 rawstride12 results on released detector tracks. They are not historical raw-t50 recertification, independent confirmation, metric/seconds, human gold, true 3D, foundation success or submission-ready evidence. Independent model selection, risk calibration and confirmation remain closed. Stage5C and SMC remain off.
