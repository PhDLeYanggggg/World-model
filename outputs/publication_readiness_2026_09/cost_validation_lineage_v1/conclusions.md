# Cost-Head Validation: What Can Actually Be Reused?

Status: completed read-only lineage audit, not new fitting or generalization.
Result source: `fresh_run` recursive exposure analysis; `cached_verified` frozen
manifests, checkpoint bytes, OOF identities and the preceding fit-audit bindings.
No future-target array member, calibration label or confirmation label was read.

## Finding

The existing OOF forecasts are excluded from their own target scene. That does
not make an all-OOF fitted cost head independently validated, nor does it make a
simple held-fold refit on the remaining OOF rows a clean validation experiment.

The experiment contract already checks recursive exposure. This audit does not
repair a missing guard or establish that an earlier validation run bypassed it.
It adds an explicit pre-fit reuse check and an explanation of why the proposed
shortcut would be rejected. The earlier fit diagnosis was correctly labelled
in-sample and remains valid as a fitting diagnosis.

## Scope And Counts

All frozen v6 Transformer/EqMotion assets for seeds 17, 29 and 43 were checked.
The two families and all seeds have the same 11,966 row identities. These are
overlapping agent windows, not 11,966 independent scenes. Repeating the same
cohort across families and seeds does not add independent observations.

| Physical fit fold | Recordings | Unique row identities |
| --- | --- | ---: |
| ETH ETH | eth_eth | 2,614 |
| ETH Hotel | eth_hotel | 1,197 |
| UCY Zara | zara01 / zara02 / zara03 | 8,155 |
| Total | Five recordings, three physical scenes | 11,966 |

Zara's three recording counts are 2,234, 5,741 and 180. They remain one physical
scene, not three independent holdouts.

| Check | Result |
| --- | ---: |
| Original producer OOF checks | 18 / 18 pass |
| Fitted cost heads | 12 |
| Fitted-head versus fit-fold validation attempts | 36 / 36 rejected |
| Hypothetical refits using other existing OOF folds | 18 / 18 rejected |
| Existing outer-held producer suitable for outer prediction | 18 / 18 pass under declared lineage |
| Outer/inner excluded-fold requirements | 36 |
| Requirements with a suitable producer in the audited pool | 0 / 36 |

The last search examines the full plus three single-held-fold forecasters within
each family/seed. It is **not** a search of every historical checkpoint in the
repository, a claim that training cannot be done, or a reason to abandon the
experiment.

## Concrete Exposure Path

Suppose ETH ETH is the outer cost-validation scene. Train a new cost head only
on ETH Hotel and UCY Zara, and recompute its normalizer on those rows. Direct
head-row overlap is now empty.

However, the cached Hotel rows came from `seed17_hold1`, and the cached Zara rows
came from `seed17_hold2`. Both forecasters declare ETH ETH in their fitting set.
Their predictions enter the cost-head features and realized cost targets.
The supposedly held validation scene has therefore already influenced the
head-training pipeline through these producers. Refitting just the last head or
normalizer does not remove that dependence.

The same issue occurs for both families, all three seeds and every outer fold.
This is an upstream validation-exposure problem, not evidence of future endpoint
features or central-velocity inputs. The original producer OOF guarantees still
pass. No new score was computed under this hypothetical refit.

## Implemented Prevention

`src/evaluation/m3w_cost_validation_lineage.py` adds a pre-fit
`require_cost_validation` entry point. It checks:

- unique fit-role validation recordings, expanded to their full physical fold;
- each head-training group's own original OOF guarantee;
- direct training-scene overlap;
- all training producers and their recursive parents against the outer holdout;
- explicitly reused preprocessing and its recursive parents;
- the producer used to predict the validation rows;
- artifact and protocol hashes before treating rejection as expected exposure.

The helper independently agrees with the existing experiment-contract guard.
An altered file is a hard error, not converted into an ordinary overlap result.
Synthetic regressions include a genuinely clean nested design that passes,
leaky preprocessors, parent exposure, renamed recordings in one scene, multiple
recordings per fold and changed artifacts. A passing preflight is consistency
with declared lineage, **not proof that declarations are complete**, IID data,
calibrated risk, or an approved scientific split.

## Smallest Defensible Next Experiment

After the outstanding scientific metric/target decision is resolved, register a
matched cost-reliability comparison before fitting:

1. Keep one outer physical scene entirely out of head training and every
   learned upstream component used for that training.
2. Within the remaining scenes, produce head-training rows with inner OOF
   forecasters that exclude both the outer validation fold and their own target
   fold. Do not warm-start from a model that saw an excluded scene.
3. Fit the candidate head and its normalizer only on those inner training rows.
   The audited outer-held forecaster can supply validation predictions, subject
   to unchanged protocol, features and verified complete lineage.
4. Compare cost reliability on held rows and intervention-eligible subsets;
   retain easy harm, cost magnitude and coverage, not just global MSE or ranking.
5. Treat this as development/model selection. It does not create independent
   risk-calibration or final-confirmation data. Previously inspected development
   scenes do not become fresh test scenes.

For the current three-fold layout there are six ordered outer/inner requirements
per family/seed. If fitting rules are identical and never selected on the outer
fold, three pair-excluded producer fits per family/seed could cover those six
requirements: 18 distinct fits across the two families and three seeds. This is
a conditional reuse count, not an approved training budget or launched run.
Only one physical training scene remains in each pair-excluded fit, an important
variance/generalization limitation. Independent data remain a separate gap.

No metric, threshold, loss, split, checkpoint or old result was changed. There
is no new prediction improvement, deployment upgrade or submission claim.
The primary-metric decision is still pending. Coordinate/time claims stay
dataset-local and annotation-step/raw-frame only. Stage5C and SMC remain off.

[Machine-readable audit](analysis.json) and [reproduction](execution_notes.md).
