# Dimensionless Features Preserve Utility, Not Safety

## What Was Run

| Component | Result source | Scope |
|---|---|---|
| SDD inputs, frozen predictors and source cost targets | cached_verified | Hash/schema/source-exclusion bindings checked |
| 72 risk-head fits, fixed decisions and new aggregate metrics | fresh_run | Full registered matrix, not a pilot-only result |
| Separate arithmetic | fresh_run | Same executor, separate implementation; not an independent researcher |
| New Transformer/EqMotion trajectory training | not_run | Intentionally frozen to isolate risk-head features |
| External prediction errors, independent calibration and confirmation | not_run | Source admission unresolved; confirmation remains closed |

All 72 registered six-output ExtraTrees fits completed: four outer SDD sites,
three seeds, three candidate actions and two feature arms. Each fit has 128 trees
and the same 768,000 source draws; unknown-label draws are zero. The predictors,
source forecasts, cost targets, cutoffs and risk budget remain frozen. Training
the risk heads is a fresh result, not new Transformer/EqMotion trajectory training.

All 175,756 source windows and 188,388 query/action/seed instances were processed.
This uses obs8/pred12 native annotation steps, stride12, annotation pixels. It is
not the historical raw t50 benchmark. Four physical sites have prior design
exposure: source exclusion at fitting does not restore independent confirmation.

## Primary Comparison

Gains below are equal-site ADE improvements over constant velocity, not a claim
that CV is the strongest causal alternative. Damping receives matched protection.

| Candidate action | Native population gain % | Dimensionless population gain % | Difference pp, nominal 95% site CI | Dimensionless worst easy degradation % | Zero-CV harms, repeated row/seed instances |
|---|---:|---:|---|---:|---:|
| Damping .05 | 1.4011 | 2.0118 | +0.6107 [-0.0457, +1.3438] | 1.2647 | 1 |
| Transformer | 3.5746 | 4.9881 | +1.4135 [+0.5718, +2.0620] | 4.4503 | 5 |
| EqMotion | 3.2976 | 5.9727 | +2.6751 [+0.7877, +5.5628] | 9.0097 | 6 |

Removing explicit native-unit inputs preserves, and here increases, aggregate
utility. It does **not** preserve the observed easy-case tradeoff for the neural
population rules. Both exceed the 2% worst-site/seed ceiling; all three actions
still harm some exact-zero CV cases. Predicted constraints passing is not evidence
that the corresponding realized risk is controlled.

The three-seed averages use 3,000 paired resamples of four physical sites. These
intervals are conditional development evidence, not independent test intervals.
They are nominal, without multiple-comparison correction. Every registered
contrast, negative result, per-site/seed metric and partial-future bound is retained
in [analysis.json](analysis.json) and [results.md](results.md).

## More Restrictive Controls

The registered dimensionless selected-denominator Transformer rule gives 2.9862%
ADE gain and 2.7917% hard gain, with 0.9437% worst positive-easy degradation and
zero observed zero-CV harms. Its gain exceeds the old strict Transformer point
estimate by 0.5493 pp, but the paired interval is [-1.0682, +1.7457] pp. The
deathCircle contrast is negative. This is a useful development signal, not proof
of a reliable improvement or grounds for readout-based policy promotion.

The equivalent EqMotion rule gains 3.8011% overall and 3.1227% on hard cases, but
its worst easy degradation is 3.5298%, so it still fails easy preservation. Damping
under this rule gains 1.3599% and has no observed easy degradation, but trails
the old strict damping rule by 2.2748 pp. The old strict damping rule itself has
2.4944% worst easy degradation. No single average-score ranking resolves safety.

## Failure Diagnosis

1. **Utility is not risk calibration.** Equal-view mean held-source easy-harm
   fraction MSE worsens from 0.06811 to 0.07978 for Transformer and 0.05404 to
   0.05519 for EqMotion. Easy-probability MSE also worsens, from 0.07710 to 0.08187
   and 0.07316 to 0.07872. These are descriptive complete-label diagnostics, not
   an independently calibrated risk test.
2. **Coverage changes materially.** Mean population selections per seed rise
   from 44,410 to 51,630 for Transformer and 28,396 to 35,819 for EqMotion. Higher
   utility alone cannot establish better decisions at identical coverage.
3. **Matched coverage is incomplete.** There are 12,569 failed exact-query-count
   controls. They fail closed and remain in the results. Their aggregate must not
   be advertised as a fully equal-coverage contrast or used to isolate ranking.
4. **The native easy definition remains.** The input arm removes explicit native
   scale, but easy membership still depends on a source-only native-unit error
   cutoff. Thus this is not an end-to-end unit-invariant safety controller. The
   worsening easy-related prediction errors motivate checking cutoff-relative
   dimensionless features, rather than loosening the 2% limit. This is a next
   hypothesis, not an established cause or a completed repair.
5. **Shared budgets permit local harm.** Overall net gain and easy-weighted signed
   risk are separate outputs. Same-query budget sharing does not imply that every
   selected easy agent is protected. The more restrictive controls expose the
   tradeoff rather than hiding it in an overall mean.
6. **Missing outcomes remain missing.** Dimensionless population policies select
   1,795 Transformer and 1,149 EqMotion unknown-label row/seed instances. Their
   outcome cannot be credited as a success. Available-grid errors and full-grid
   bounds remain distinct; the observed easy subset is not the whole population.

The solver records 941,940 policy-query calls, zero reported original-unit risk
violations, and 12,575 instances without proved optimality, including the 12,569
failed count controls. Of those count failures, 1,369 have insufficient eligible
support; 11,200 carry the generic solver-nonoptimal-floor status, which does not
distinguish infeasibility from other unsuccessful solver terminations. Six other
original-unit-check cases fail closed. Exhaustive global optimality is not
established; a feasible empty fallback is not proof that an optimum was found.

## What Changes and What Does Not

The evidence supports retaining a dimensionless risk-head research route. It does
not support promoting the highest-gain population policy, certifying safety,
claiming neural necessity, or declaring a cross-domain world model. The selected
Transformer control warrants a new predeclared study, not retroactive selection
as this experiment's winner.

The separate IMPTC precision repair passes its fixed numerical probes, but was
not mixed into this training experiment. IMPTC outcomes remain unread and its
source remains quarantined. DroneCrowd confirmation remains closed. No deployment,
Stage5C, SMC, metric/seconds, true-3D or foundation-model claim changes.

Next steps are to keep the current forecasts and easy criterion fixed, test
cutoff-relative risk features on source-only folds, and resolve independent
calibration-source admission before any external risk claim. A fixed source
cutoff in pixels cannot silently become a calibrated threshold for another
dataset's local coordinates.

## Evidence

- [Frozen registration](registration.md), [method](method.md) and [operation guide](operation_zh.md).
- [All policy results](results.md) and [all training losses](training_losses.md).
- Full decision/aggregate replay is exact, and separate arithmetic passes. The
  receipts and [execution record](execution_notes.md) state their scope and limits.
- [Tradeoff figure](risk_tradeoff.svg) retains all six registered feature/policy
  controls and old strict; omitted uncontrolled/count-matched arms remain in the table.

Verdict: **source utility improved; neural population easy preservation failed;
no policy promotion and not submission-ready.**
