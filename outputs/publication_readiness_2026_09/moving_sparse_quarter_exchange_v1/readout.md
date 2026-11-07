# Learned Sparse Exchange: Capacity Does Not Transfer To Decisions

**Decision:** Retain the original quarter roster. The learned method
does not recover the improvements found by the label-aware search.

## What Changed

I trained a paired cost model on the change in utility, risk and
easy-reference support when exchanging two agents in one query. Both
rankings share the same network. The support variant adds a score for
retaining or gaining an easy reference; the risk-only variant omits it.
Each policy takes at most the first four qualifying exchanges in
chronological order, one per query. None changes the original intervention
count or quarter amplitude. No saved hindsight choices enter training
or inference. Donors select thresholds independently for each fold.

## Results

| Policy | Pass / 102 | Total Risk >2% | Signed Easy >2% | Undefined Easy | Agent Replacements |
|---|---:|---:|---:|---:|---:|
| Original quarter roster | 15 | 39 | 0 | 35 | 0 |
| Dense gain/harm ranking | 7 | 56 | 2 | 35 | 6,766 |
| Sparse risk, threshold-0 diagnostic | 16 | 46 | 1 | 33 | 301 |
| Sparse support, threshold-0 diagnostic | 16 | 47 | 2 | 31 | 302 |
| Sparse risk, donor-calibrated | 15 | 40 | 1 | 33 | 18 |
| Sparse support, donor-calibrated | 15 | 41 | 1 | 34 | 19 |

| Paired Contrast | Completion Lower (pp) | Source Bootstrap 95% CI (pp) |
|---|---:|---:|
| Calibrated support vs original quarter (primary) | +0.00006 | [-0.00224, +0.00242] |
| Calibrated risk vs original quarter | +0.00127 | [0, +0.00312] |
| Threshold-0 support vs threshold-0 risk | -0.01541 | [-0.02933, -0.00375] |

The original 6,952 interventions are unchanged in count. The 102 views
are 34 exposed development recordings across three forecaster seeds
and 14 source groups. The comparison between threshold-0 arms is
matched on query intervention count, not exactly on the number of
replacements, which differs by one overall.

A descriptive recount of individual transitions distinguishes equal
totals from equal outcomes. The threshold-0 support arm gains 3 passes
and loses 2 relative to quarter; it gains a defined easy reference in
5 views and loses one. The risk-only probe gains 2 passes and loses 1.
Neither donor-calibrated variant adds or loses any passing view.
The support variant falls back on 38 of 42 folds, recovers a reference
in one held view, but does not make it pass. The sparse strategy avoids
many of the losses of dense reranking, but that is not improvement over
the original roster. Signed-easy violations remain visible.

## What This Means For The Paper

The capacity result with labels and the past-only result support
different claims. One hindsight replacement repairs 26 failing views,
but the learned, donor-selected policy does not repair any held view.
This is evidence against the current paired scoring and admission
rule, not proof that the candidate bank has no good alternatives.
Regression deferral already provides general formulations for choosing
among predictors ([Mao et al.](https://proceedings.mlr.press/v235/mao24d.html)).
Here the scientific test is whether paired edits retain empirical
reference support and utility under the recording rule. Replacing
agents jointly in the roster does not by itself establish physically
consistent interaction forecasts; that calls for separate joint
metrics ([Weng et al.](https://arxiv.org/abs/2305.06292)).

## Reproduction

```sh
.venv-pytorch/bin/python scripts/run_m3w_sparse_quarter_exchange.py fit
.venv-pytorch/bin/python scripts/run_m3w_sparse_quarter_exchange.py score
.venv-pytorch/bin/python scripts/run_m3w_sparse_quarter_exchange.py replay
```

Registered at `4899b9fd`. All 42 native arm64 Torch fits completed
440 updates each (3,297,125 sampled pair occurrences across repeated
folds). Four compute threads, no loader workers, checkpoint every 55
updates. Fit took 177 seconds; score and replay about 13 seconds each.
Fourteen scoped tests pass. Both controls and the [analysis](analysis.json)
replay exactly. Data and fitted checkpoints remain local.

The evaluation uses 8 observed and 12 predicted native steps in image
coordinates. It is not independent confirmation or a claim about
metric distance, seconds, true 3D or foundation modeling. No new
deployment or external outcome access. Stage5C and SMC stay off.
