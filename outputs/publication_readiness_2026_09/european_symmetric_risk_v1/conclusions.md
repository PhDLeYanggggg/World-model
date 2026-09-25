# Symmetric Risk: More Neural Gain, Less Reliable Protection

## Decision

The experiment is complete and reproducible, but the proposed safety repair
fails. Replacing asymmetric risk regression with MSE increases neural ADE gains
in the easy-event policy family while violating the 2% worst-locality easy
limit in every seed. It does not establish stable neural superiority over an
equally protected causal damping control. Deployment is unchanged.

This is a source-development result, not independent confirmation. All 48
registered views are retained in [results.md](results.md), including failures.
No view, checkpoint, seed or threshold was selected from these outcomes.

## What Changed

36 new Torch event-risk heads, three folds and three seeds, 72,000 optimizer
updates. Both the frozen neural forecast and damping 0.97 receive the same
single-factor change: underharm4 risk loss to symmetric MSE. The preceding
18 symmetric-utility heads, trajectories, 355 causal features, labels,
preprocessing, draws, initialization constants and 2% predicted-risk budget
remain fixed. The 100-update runtime pilot resumes within the total budget.

All 36 fits finished before outcome readout. The 24 ridge-risk policy views
are cached_verified controls with exact old decisions, not new optimization.
No trajectory model, JEPA encoder or end-to-end world model was retrained.

## Accuracy and Safety

Fixed easy-event neural-risk views without the source-support guard:

| Seed | Old neural ADE gain vs CV | New neural ADE gain vs CV, conditional 95% CI | New worst-locality easy degradation | Zero-CV cases harmed | New protected damping ADE gain vs CV |
|---|---:|---|---:|---:|---:|
| 17 | 0.2719% | 1.1341% [0.6461%, 1.6465%] | 2.4644% | 0/4 | 1.6711% |
| 29 | 0.1982% | 1.5810% [0.6950%, 2.8064%] | 7.2891% | 1/4 | 1.1509% |
| 43 | 0.4188% | 2.0901% [0.6028%, 4.0246%] | 16.7135% | 0/4 | 1.8723% |

The direct neural new-versus-old gains are 0.8655%, 1.3884%, 1.6882%, each
with a positive conditional interval. Those gains are not free: all six new
easy-event neural views fail observed safety, including the guarded variants.
The guard removes the seed29 zero-CV harm but not the worst-locality failure.
All six new all-event neural views preserve full-population observed safety,
but their unguarded ADE gains are only 0.2730%, 0.2544%, 0.6490%.

Among the 12 new-risk neural-versus-damping pointwise comparisons, four point
estimates are positive but no interval is strictly positive; seven are strictly
negative. Two hard-subset intervals favor neural, but both belong to seed29's
unsafe easy-event family. They are correlated views, not independent replication.
The causal control is refitted fairly: all 12 new damping views pass full
observed safety, although its easy-event accuracy falls versus the old risk loss.

Across all 48 views, full observed safety passes 16/24 neural and 21/24 damping
views. Exact-count joint safety passes 5/24 and 20/24 respectively. Among new
risk views alone, neural joint safety is 0/12 and damping is 11/12.

## Why the Repair Fails

The old neural easy-event heads overestimate population-average positive harm.
MSE crosses to underestimation instead of producing calibrated risk:

| Seed | Old predicted event harm | New predicted event harm | Realized event harm |
|---|---:|---:|---:|
| 17 | 0.4610 | 0.0986 | 0.1782 |
| 29 | 0.4228 | 0.0937 | 0.1776 |
| 43 | 0.3651 | 0.0756 | 0.1832 |

These are equal-locality means of positive easy-event harm in image-pixel ADE
units. A lower average regression bias does not establish reliability on the
rows selected for intervention. The largest easy degradation occurs in
eu-locality-110 in all three seeds:

| Seed | Selected ADE-supported rows | Predicted selected event-harm ratio | Realized selected event-harm ratio |
|---|---:|---:|---:|
| 17 | 675 | 1.2711% | 55.9529% |
| 29 | 1,913 | 0.8434% | 70.0238% |
| 43 | 3,408 | 0.4382% | 65.6131% |

These ratios divide positive easy-event harm by easy-event CV-error mass on
selected supported rows. They are NOT net easy degradation percentages and
are not computed on the same denominator as the safety gate. Non-easy rows
have zero event targets. Zero event mass remains undefined, not zero risk.
The large mismatch shows severe selection-conditional risk underestimation.
It does not by itself identify whether feature limits, event imbalance,
producer-training shift or finite fitting capacity is the underlying cause.

The loss experiment therefore supports a narrow conclusion: earlier risk
conservatism rejected useful interventions but also prevented real harm.
Relaxing a loss or improving population mean error is not a safety solution.
The complete [reliability table](risk_reliability_table.md) preserves all
localities, selected/unselected support and old/new versions.

## Joint Control and Numerical Failures

One matched-count joint-versus-unary contrast is positive: damping seed17,
all-event MSE risk, unguarded, 0.00669% [0.00036%, 0.01550%]. This tiny
single-seed result does not establish a stable coordination contribution.
Neural seed43 all-event unguarded also has a positive ordinary joint-versus-
independent interval, 0.00972% [0.00301%, 0.01756%]; that comparison does not
match intervention counts. It cannot replace the matched-count test.

The inherited ridge-control solve for recording98/frame17090, 21 agents,
returns an invalid-solution safe floor. It remains failed/unmatched, not
optimal, and was not rerun. Every new neural-risk solver call succeeds.
The joint pilot contains no zero-CV cases and cannot validate their protection.

## Verification and Limits

- Full 48-view metrics reproduce; all 36 new checkpoints reproduce 4,096 rows
  exactly and match old draws, preprocessing and sampling RNG.
- All 144 pointwise decision records independently reconstruct. Joint predicted
  budgets and within-candidate matched counts pass accounting. Equal budgets
  across candidates do not imply equal actual intervention counts.
- 185 tests in 25 scoped files pass. This is not the full legacy test suite.
- 318,969 targets, including 7,047 future-unknown targets retained for inference;
  311,922 ADE-supported and 240,269 FDE-supported rows. Unknown targets never
  enter supervised draws. Joint pilot: 1,152 queries / 6,116 targets.
- Three seeds and 3,000 locality-bootstrap resamples are conditional on the
  opened source-development data and fitted models. Overlapping rows and these
  correlated configurations are not independent observations or confirmation.
- Twelve opened source localities only. Reserved model-selection, calibration,
  confirmation and DroneCrowd remain closed. Historical Stage35/37 exploratory
  results are not re-certified by this experiment.
- Released detector tracks, image pixels, obs8/pred12 at raw stride12. This is
  not t50, metric, seconds-level, human gold, physical safety, true3D, foundation
  success or CCF-A submission readiness. Stage5C and SMC remain disabled.

## Next Experiment

Diagnose selection-conditional reliability and producer shift using source-only
nested predictions, then register a source-only calibration intervention for
both candidates. Freeze the score model before fitting a calibration rule and
retain a separate outer locality readout; do not tune on the outcomes above or
open reserved roles to rescue this result. Compare discrimination as well as
mean calibration. If the head cannot rank safe neural opportunities, calibration
alone will only return it to fallback; candidate forecasting and context then
need a controlled improvement. No such repair is claimed completed here.
