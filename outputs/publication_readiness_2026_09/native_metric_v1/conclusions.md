# What the Native-Coordinate Amendment Changes

## Decision

The author delegated the choice of research route. We adopted native-coordinate
ADE/FDE and equal-scene relative ADE gain because the goal is overall motion
forecasting, not almost exclusively predicting departures from static histories.
This is a documented post-hoc evaluation amendment, frozen in commit d1693296
before the new readout. It is not a guarantee of stronger results or publication.
The original metric, source audit, negative studies and immutable protocols were
not rewritten. Their historical "pending" flags describe their original dates;
the latest decision is now resolved.

## Fresh Results, Verified Cached Inputs

The readout retains all 175,756 past-eligible queries from 33 recordings of four
previously explored SDD source sites. There are 172,957 queries with at least one
future label, 143,918 with complete futures, and 2,799 without any future label.
Unsupported labels remain unknown. No original validation/test, bookstore or
main/external labels were opened.

| Fixed comparison | Supported masked ADE gain vs causal CV | Complete-future sensitivity |
| --- | ---: | ---: |
| Constant velocity, causal finite difference | 0.00% | 0.00% |
| Damped velocity, coefficient 0.05 | -6.62% | -7.64% |
| Other-source-sites-selected fixed baseline | 0.00% | 0.00% |
| Per-query oracle, diagnostic only | +28.63% | +28.35% |

All seven causal baselines are retained in [the complete tables](results.md).
The source-complement selection rule chooses CV in every fold. Oracle results
use future labels and cannot be deployed. The oracle's exploratory scene
bootstrap interval is [25.94%, 32.56%] for supported masked ADE and
[25.68%, 32.64%] for complete futures. These intervals concern oracle headroom,
not a learned improvement. Four reused sites and overlapping training histories
do not constitute independent confirmation.

For context, the corresponding old normalized oracle headroom was approximately
0.0395% / 0.0383%. These are different weightings of the same source population,
not before/after model gains. The new weighting makes a different part of the
forecasting problem visible. It does not make oracle information causal.

| Physical scene | Supported rows | Causal CV native ADE, annotation pixels |
| --- | ---: | ---: |
| coupa | 27,778 | 11.9082 |
| deathCircle | 35,760 | 24.8107 |
| gates | 20,236 | 19.7615 |
| hyang | 89,183 | 15.9234 |

Native units are not comparable physical distances across scenes. The primary
summary averages within-scene relative gains, not these raw pixel values.

## Neural Negative Result Retained

The existing source-crossfit cache contains all twelve fixed fits across three
seeds, but covers only 15,430 static-history, complete-label queries. It cannot
stand in for a full-population neural experiment. Re-scoring those exact cached
predictions yields:

| Seed | Native equal-scene ADE gain vs CV | Old normalized gain |
| --- | ---: | ---: |
| 17 | -5.4579% | -5.0831% |
| 29 | -6.1374% | -5.7094% |
| 43 | -4.5496% | -4.2555% |

The mean-per-seed-error result is -5.3816%, not an ensemble forecast. Changing the
metric therefore does not repair this neural failure. Zero-CV easy cases retain
positive absolute pixel harm; a relative percentage there is undefined, not a
safety pass. No seed, model or threshold was selected from these results.

## Verification

- 198 original source arrays verified against the prior audit.
- 1,230,292 baseline/query ADE/FDE pairs reconstructed from paired trajectories.
- All three neural caches matched to the original IDs and supervision, with the
  twelve checkpoint/parent/prediction receipts verified without executing them.
- Exact completed re-run reproduces the analysis and generated table.
- Independent scalar reduction verifies 62 metric tables, 248 scene reductions
  and the 3,000-resample intervals; 267 dependency hashes checked.
- 66 scoped tests pass: new metrics, old source population, conditional-risk
  identities and experiment-contract tests. No full historical suite rerun.

See [machine-readable analysis](analysis.json) and [independent verification](verification.json).
Analysis SHA256: `df496c182dfff57c27dee585606d23613341de3e9946fb9786c1c12ce3c32969`.

## Shortest Next Experiment

Register a matched full-population source comparison under native ADE before new
fitting: causal CV, train-selected baseline and existing model families with
identical membership, masks and training budgets. Separate train-only numerical
conditioning from the actual scientific loss. Do not reuse a static-only cache
as full-population coverage, or blindly repeat all old fits.

For selection, use genuinely outer-held cost evaluation and nested producer
exclusion. Register native-scale easy/tail harm safeguards before threshold
selection. Better aggregate cost fit is not conditional intervention reliability.
Preserve static starts as a named failure slice, not an excluded population.

No new training or inference occurred in this readout. No new independent
calibration or confirmation was obtained. No deployment or submission-candidate
claim is supported. SDD remains annotation-pixel/raw-frame; Stage5C and SMC remain
disabled. The broader research goal remains active and unmet.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/rescore_m3w_native_metric.py --verify
.venv-pytorch/bin/python scripts/verify_m3w_native_metric.py --verify
.venv-pytorch/bin/python -m pytest tests/test_m3w_native_metrics.py tests/test_m3w_source_population.py tests/test_m3w_conditional_risk_identities.py tests/test_m3w_experiment_contract.py -q
```
