# Failure Taxonomy and Next Test

## Confirmed Findings

1. **Primary estimand feasibility error.** The matched-count parent has ten
   zero-action views. Selected-harm ratios are undefined there before any new
   model can change the outcome. The full-roster primary cannot succeed. This
   is our design error; it cannot be blamed on training, hidden by deleting
   rows, or repaired retrospectively by switching denominators.
2. **No stable same-count accuracy gain.** The registered secondary matched
   ranking contrast is +0.009026%, with a locality CI crossing zero. Seven of
   twelve locality contrasts are nonpositive. The risk-ranking arms have 95
   versus 89 violating dependent views at the same count, with ten undefined
   views each. The fixed-roster aggregate risk contrast remains undefined.
3. **Joint performance deteriorates.** Query supervision loses 0.154680% ADE
   relative to pointwise supervision at the same nominal budget. It intervenes
   0.7210 percentage points less. This contrast does not isolate ranking from
   coverage, which is why the count-matched contrast is also retained.
4. **Fitting improvement does not transfer.** Own-objective fitting loss falls
   in 108/108 pointwise and 104/108 query heads. Nevertheless held-query MSE
   point estimates are worse under query supervision: +1.32% all and +3.61%
   easy, compared with the pointwise model on the same aggregate metrics.
   These relative changes are descriptive; no extra significance test is added.
5. **Net easy gain is insufficient.** Query joint preserves easy net error and
   exact-CV cases, yet 98/216 dependent views exceed selected positive-harm 2%.
   Positive and negative trajectory changes cannot cancel in that harm screen.
6. **Missing outcomes persist.** Unknown-label rows remain eligible for causal
   inference. Query joint averages 29.93 unknown interventions per dependent
   locality view. Their actual harm is not observed and cannot be certified.

## Mechanisms Consistent With the Code, Not Proven Causal Explanations

- Pure aggregate squared error permits an overprediction for one agent to
  cancel an underprediction for another. The regression test demonstrates the
  mathematical possibility, not how much of the observed failure it causes.
- Training uses every known fitting agent in a sampled current query. Deployment
  evaluates an adaptively selected subset. Accurate full-query sums do not
  identify risk on arbitrary chosen subsets.
- Four positive output bases are trained through only two signed combinations.
  They must not be interpreted as four calibrated cost moments.
- A joint solver can exploit errors in predicted slack. Solver feasibility is
  checked, but that checks predictions, not realized risk or physical safety.
- Singleton queries contribute no aggregation contrast. Their held proportion
  is about 29.96%; this alone does not explain the negative result.

No evidence here establishes that larger networks, more training, or new
scene modalities are the necessary fix. Those factors were deliberately held
constant. The present test rejects this particular pure-aggregation repair,
not every possible query-aware method.

## Shortest Next Repair

Before fitting another variant, preregister a coverage-aware evaluation whose
diagnostic denominator is defined under abstention. Keep the original selected
positive-harm ratio explicitly undefined when no selected reference error
exists; do not weaken its 2% safety screen or manufacture intervention coverage.

Then test selected-subset excess supervision against the matched pointwise
control, using subsets generated causally on fitting sources and retaining an
individual-error anchor. Freeze the subset generator and the new objective
before fitting, with no held-out threshold sweep. This targets a concrete
training/deployment mismatch while keeping features, predictors and source
roles fixed. It is a proposed next experiment, not an executed repair or claim
of guaranteed improvement.

Only twelve development localities were read; independent roles remain closed.
Image-local detector silver, obs8/pred12 raw-frame stride12. No metric, seconds,
human-gold, physical-safety, true3D or foundation claim. No Stage5C or SMC.
