# Stationary Starts: Source-Verified, Not Yet Cross-Scene Predictable

## What Was Run

This is a fit-only diagnostic under the frozen eight-observed/twelve-predicted
native-step protocol. The forecasting task, target rows, primary metric and
raw50 supplement were not changed. No development, calibration or confirmation
labels were opened for this experiment. Existing data and producer identities
were verified; the source replay, 48 classifier fits and saved-model prediction
replay were run afresh. No forecasting network or deployed policy was changed.

The first extraction stopped on a NumPy-boolean JSON serialization error before
any classifier was fitted. The failed registration and source snapshot are
preserved. The versioned repair changes only serialization. It does not rewrite
the failed experiment's hashes or pretend that it completed.

## Source Evidence

Every cached coordinate row in the five fit recordings matches the canonical
cleaned source positions exactly. There are no malformed source rows or gaps in
the stationary histories inspected here. This rejects cache corruption as an
explanation of these particular stationary cases; it does not verify annotation
construction, image synchronization, physical stillness or causal sensor access.
The previously found ETH clock conflict is still unresolved.

Exactly stationary eight-step histories occur in 365 of 11,966 fit windows:
81 ETH windows and 284 Hotel windows. They cover only 31 agents and 45 stationary
runs at two physical sites. Zara contributes no stationary windows and its
held-fold probe is explicitly not_run. The 365 overlapping windows are not 365
independent starts. The label is any recorded coordinate change in the next
twelve requested steps, not a verified physical start or intention annotation.

The changed-future rate is 59/81 (72.84%) in ETH and 129/284 (45.42%) in Hotel.
Thus both small support and a substantial scene-associated prevalence shift
must be considered. The prior geometric audit attributed 73.25% of pooled fit
CV error to these windows under the unchanged normalization; that is not their
share of independent events or of the equal-scene development primary error.

## Initial Hypothesis and Repair

The initial hypothesis was that past neighbor configuration or motion would
predict recorded movement after an otherwise uninformative stationary ego
history. Inputs exclude absolute location, scene identity, future availability,
future run boundaries, supplied central velocities, teacher outputs and goals.
Only neighbors with complete histories aligned to the same eight observed
steps are admitted. The baseline is the opposite-scene training-only smoothed
change rate; a stationary ego history alone has no varying motion signal.

We fitted logistic regression and ExtraTrees with fixed settings, two feature
sets and seeds 17/29/43, holding out ETH or Hotel by physical scene. All 24 ordered
feature models were worse than the train-only prior in window-level Brier score.
Deterministic logistic seeds give identical fits, not independent replications.

One adaptive repair replaced ordered neighbor slots with permutation-invariant
summaries: 19 to six geometry features, or 83 to thirteen geometry/motion features.
The rows, folds, labels, model settings and seeds stayed fixed. This tests whether
high-dimensional ordering is part of the problem, without retuning thresholds.
It was chosen after the initial failure and is not independent confirmation.

| Direction / pooled geometry ExtraTrees | AUC, seed mean | Brier improvement vs train prior | Run-balanced improvement | Agent-balanced improvement |
| --- | ---: | ---: | ---: | ---: |
| Hotel train -> ETH held | 0.6949 | +0.02513 | +0.02012 | +0.04701 |
| ETH train -> Hotel held | 0.4895 | -0.01943 | -0.00285 | +0.00494 |

These are absolute Brier-score differences, not percentages or trajectory gains.
The first direction improves window-level Brier in all three tree seeds; it is
the only positive setting among the 24 repaired fits. Its held support is only
five agents. Reverse transfer fails, and adding motion summaries does not give
positive window-level Brier in either direction. Reweighting changes some signs,
so every window/run/agent result is retained rather than choosing favorable ones.

## What This Explains, and What It Does Not

1. Pure ego-motion bounding cannot recover starts from exactly stationary pasts.
   That is a construction limit, not something more routing thresholds can fix.
2. Ordered neighbor features fail this probability-prediction test; pooling
   recovers one local signal but does not establish robust bidirectional transfer.
3. More context is not automatically better. The tested motion summaries did not
   improve Brier over the opposite-scene prior, despite occasional ranking lift.
4. Sparse agent/run support and differing change rates make cross-scene learning
   fragile. Their individual causal contributions are not isolated by this study.
5. Predicting a recorded start is easier than predicting its direction and
   displacement. No trajectory improvement follows from this classifier result.

This does not prove that stationary behavior is irreducible, that all interaction
models fail, or that annotation artifacts caused the original forecasting error.
It also does not justify removing stationary rows, changing the primary metric,
calibrating against the held labels, or claiming a new world-model contribution.

## Decision and Next Experiment

Do not launch a stationary-start residual head based on this evidence. Retain
the current frozen forecasting/policy results and keep this diagnostic separate
from the paper's primary comparison. No new model is deployed.

The next useful investigation is a fit-only availability/identifiability audit
of scene cues and relative start directions. It should first determine whether
legally usable scene context can be aligned without the unresolved ETH clock
assumption, and whether the same eight-step input has enough independent agent
and site support. Any direction probe should retain whole-scene exclusion and
compare against train-only priors before authorizing another neural forecast.
If support is insufficient, document the data requirement rather than repeatedly
selecting among the same two sites. Do not expand observation length silently.

Independent source-eligible confirmation sites, stronger candidate forecasts,
and evidence for the proposed joint intervention mechanism remain open. These
results can support a limitations/diagnostic section, not a positive main claim
or an assertion that the project is ready for an A-venue submission.

## Verification and Boundaries

All 48 saved classifiers replay their reported Brier scores within 1e-12.
Window, run and agent summaries are descriptive, with no independent-scene CI
or significance claim. Seven focused tests passed, including future-input
rejection, scale/rotation invariance, pooling invariance, run serialization and
group-weighting behavior. The unchanged legacy full suite was not rerun.

Source units remain dataset-local; no verified seconds, metric, physical safety,
true-3D or foundation result is claimed. Stage5C and SMC remain disabled.
Code, registrations and aggregate metrics are versioned; source data, row caches
and classifier checkpoints remain local and excluded from Git.

See [all outcomes](results.md), [original metrics](metrics.json),
[pooled metrics](pooled_metrics.json), [saved-model replay](comparison.json),
[initial failure](extraction_failure.md) and [repair decision](pooled_repair_decision.md).
