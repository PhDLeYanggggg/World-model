# Early Query Weighting Hurts Within-Query Easy-Reference Ranking

The previous agent classifier showed an average ranking signal but
missed easy reference at the first eligible query. Here the target
is the probability mass assigned to all observed easy agents
within one query. The two PyTorch arms use the same 24-unit head,
features, initialization, query draws, and 220 updates. Only the
weight of mixed queries among the first two eligible queries
changes, from one to eight. Both evaluate the exact same eligible
agents, including queries that were homogeneous and therefore
uninformative for listwise fitting.

| Score | First eligible query | Mean of first two | All eligible queries | First hits in 35 missing-support views |
|---|---:|---:|---:|---:|
| Frozen joint easy/safe | 33.81% | 30.83% | 34.88% | 6 |
| Frozen pointwise boosted | 32.46% | 29.84% | 39.01% | 5 |
| Uniform listwise | 31.90% | 30.32% | 34.24% | 7 |
| Early-weighted listwise | 20.56% | 19.37% | 28.98% | 4 |

For the registered primary contrast, early weighting loses
**11.35 pp** against uniform listwise (95% paired source interval
**[-23.02, -2.06]**). Its first-query effects against frozen boosting
and joint scoring are -11.90 pp [-24.60, 0] and -13.25 pp
[-31.51, +3.26], respectively. Early weighting also loses
10.95 pp [-20.08, -3.41] on the first-two-query average versus
uniform weighting. None advances the registered component screen.

The results average recordings within source and seed, then average
the three seeds within each of 14 source groups. Intervals use 3,000
paired source resamples. The 102 views repeat 34 recordings.

## Failure Pattern

The first-query paired diagnostic loses 11 previously correct
uniform-model views and regains one; 21 remain correct. Five
sources worsen and nine tie, with no source improving on this
comparison. In the 35 missing-support views, 28 have at least one
easy agent available. Uniform ranking hits seven of the 35; early
weighting retains four of those hits and adds none. This is a
failure of the tested objective weighting and input representation,
not an identified mechanism of overfitting. More emphasis on
early supervision does not fix the ranking under this design.

Listwise ranking itself is established (see
[Cao et al., ICML 2007](https://doi.org/10.1145/1273496.1273513)).
This experiment uses a set-of-positive-agents objective to test
early reference selection, not a new general ranking method.
No score is used to change an operating policy.

## Reproduction

Registration `6ae0542c` precedes 84 native arm64 Torch CPU fits,
each with 220 updates, four compute threads and no loader workers.
Mixed-query training has 32,160 repeated fold-query occurrences,
1,260 early, not independent sample counts. The fitting-source
normalization, model and optimizer are saved; checkpoints occur
every 55 updates. With the existing private banks available:

```sh
.venv-pytorch/bin/python scripts/run_m3w_early_reference_listwise.py fit
.venv-pytorch/bin/python scripts/run_m3w_early_reference_listwise.py score
.venv-pytorch/bin/python scripts/run_m3w_early_reference_listwise.py replay
.venv-pytorch/bin/python scripts/audit_m3w_early_reference_transitions.py
```

Score replay reproduces the aggregate hash, not a new training run.
The transition audit uses exposed labels without altering scores.
Native image coordinates, eight observed and twelve forecast
annotation steps. No independent outcomes, 2% rule changes,
deployment, metric/seconds claim, Stage5C, or SMC.
