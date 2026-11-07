# Observed Easy Reference Is Partly Predictable, But Early Ranking Still Fails

The previous ordered joint policy proposed insertions without
acquiring observed easy reference. Here I isolate one question:
can a score rank agents with a positive observed causal-floor
error within the original easy cutoff? The label is agent-level,
not a property of a specific alternative forecast. Wholly
unknown outcomes provide no observed reference; this does not
classify their true error.

The exact same eligible agents are used for every score.
Features use past motion, same-query neighbors, and causal
summaries from source-excluded heads. Ridge and fixed-depth
histogram boosting share these features. The two existing
controls are the best predicted easy mass and best joint
easy/safe score across legal families. The study fits 84 heads
in 42 cells, without tuning on held outcomes.

| Score | Top-one easy rate, all queries | Top-one easy rate, first query | Pooled weighted AUROC | Pooled weighted AP |
|---|---:|---:|---:|---:|
| Predicted easy mass | 32.82% | 27.14% | 0.706 | 0.437 |
| Joint easy/safe | 34.88% | 33.81% | 0.650 | 0.411 |
| Ridge | 34.88% | 24.05% | 0.687 | 0.466 |
| Boosted | 39.01% | 32.46% | 0.715 | 0.522 |

The 369,668 eligible agent-query occurrences and 4,839 query
occurrences repeat 34 recordings across three forecaster
seeds; they are not independent. Pooled weighted prevalence is
0.220. For top-one rates, recordings are averaged within source
and seed, then seeds within each of 14 sources. The paired
95% intervals use 3,000 source resamples.

Across queries, boosting improves by +6.19 pp versus predicted
mass [0.02, 12.14], but only +4.13 pp versus the stronger
top-one joint control [-1.78, 10.25]. At the first eligible
query the latter contrast is -1.35 pp [-23.02, 17.78].
The required positive interval versus both controls is not met.
This score is not promoted to a joint policy.

## Early Failure And Candidate Capacity

At the first eligible query of the 35 original missing-support
views, 28 have an observed easy agent available. The boosted
winner hits five views and misses 23 with an available easy
agent; the old joint winner hits six. Of the boosted model's
30 misses, 29 have observed floor error above the easy cutoff
and one has a wholly unknown outcome. Early failure is mostly
misranking an available agent, not absence of labeled support.

At the same eight-action budget, boosted retrieval retains
21/35 checked ordered hindsight repairs, ridge retains 10/35,
the previous support-specific list retains 20/35, and the
full bank 33/35. These are label-aware capacity checks, not
online joint-policy results. No outcomes tune the models or
change the frozen decisions. Stronger pooled discrimination
alone does not justify intervention at the first query.

## Reproduction

Registration `73cb417f` precedes all fits. From the repository
root with the existing private forecast and fold artifacts:

```sh
.venv-pytorch/bin/python scripts/run_m3w_easy_reference_rank.py fit
.venv-pytorch/bin/python scripts/run_m3w_easy_reference_rank.py score
.venv-pytorch/bin/python scripts/run_m3w_easy_reference_rank.py replay
.venv-pytorch/bin/python scripts/audit_m3w_easy_reference_first_query.py
```

The fits resume at completed source checkpoints, and the scoring
and capacity replay reproduces the aggregate hash. The first-query
failure diagnostic uses already-exposed labels without changing
any model decision. Moving A native image coordinates and
eight observed/twelve predicted annotation steps only. No
operating policy, deployment, independent confirmation, metric
or seconds claim, physical safety, Stage5C, or SMC.
