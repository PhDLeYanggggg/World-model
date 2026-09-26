# Nested Residual Supervision: No Promotion

## Verdict

The registered locality-OOF residual repair does not improve the original
cost estimator reliably. Both the primary comparison gate and the tail/coverage
guard fail. Keep the original estimator and deployment unchanged. This is a
completed negative risk-estimation experiment, not a trajectory improvement,
an independent confirmation result or a submission-ready method.

## What Was Actually Run

- Fresh training: 432 native-arm64 Torch risk heads, 864,000 optimizer updates.
- Fresh fixed residual fitting: 864 global/context ridge probes, no threshold search.
- Fresh source-development readout: all 36 groups / 144 views / three seeds.
- Cached_verified controls: frozen trajectory producers, original outer risk
  estimators and the prior three-locality in-sample correction.
- Not_run: new trajectory training, policy utility, independent selection,
  reserved calibration, confirmation, Stage5C and SMC.

The original registration is retained with three implementation-only amendments:
Boolean JSON serialization, truthful OOF provenance metadata and receipt-path
loading. Each is hash-bound and documented. No amendment changes the scientific
conditions, coefficients, predictions or metric calculations. Inner prediction
freeze 9259a611 precedes residual fitting; prediction freeze 94442ac2 precedes
the new outer readout. The full run did not substitute a pilot for the budget.

## Primary Evidence

Positive means lower easy-harm MSE on known positive-disagreement rows.
The range below is the range of six point estimates, not one confidence interval.

| OOF context repair compared with | Inputs | Positive / negative / overlapping 95% intervals | Point-estimate range (%) |
|---|---|---:|---:|
| Original outer estimator | Full | 0 / 0 / 6 | -5.0019 to +0.1976 |
| Prior in-sample context repair | Full | 0 / 1 / 5 | -2.9367 to +1.1441 |
| OOF global-bias repair | Full | 1 / 0 / 5 | -3.7729 to +0.3416 |
| Matched cyclic next | Full | 0 / 1 / 5 | -4.9829 to +0.3961 |
| Matched cyclic previous | Full | 0 / 0 / 6 | -3.6224 to +0.2918 |
| Original outer estimator | Motion-only | 0 / 0 / 6 | -22.0430 to -3.2028 |

Each contrast averages three seeds within locality, then uses 3,000 paired
resamples of four localities. The six assignments overlap; these are not six
independent studies or multiplicity-adjusted claims. Wide intervals are not
proof of equivalence. A favorable secondary contrast cannot replace the
registered original-estimator comparison. The coverage guard fails against
the previous in-sample context correction in one full-input assignment.

## What The Experiment Rules Out

Simply changing the residual bank from training-internal to locality-OOF
predictions is insufficient to repair this fixed seven-feature additive
correction. The experiment does not show that OOF methods are generally
ineffective, or that the context features cannot help another well-specified
task. It also does not isolate exposure from changed training composition
and event-cut transport.

Inner/outer easy-event disagreement has a 2.5245% median for OOF, versus
0.9878% and 1.2659% for the two cyclic controls. The OOF maximum is 15.0120%.
These are fitting-bank label disagreements, not test errors or a proven cause
of the failed repair. Minimum easy-harm support is 72 rows for full input and
five for motion-only, with correlated rows and few localities.

The median fixed-fitting-batch loss falls from 0.651791 to 0.433402. That
confirms optimization, not useful cross-locality correction. No trajectory
output is changed; ADE/FDE gain, physical safety and deployment gain are not
measured by these cost-head results.

## Follow-Through

The [error accounting](residual_error_accounting.md) separates aggregate
direction mismatch from excessive correction magnitude without fitting a
held-optimal coefficient. Full verification must replay every head and probe,
all readout groups and independent MSE arithmetic; its completed status is
recorded separately in `verification.json`, not inferred from this narrative.
The next repair must address an observed transport mechanism, retain the
strong original control and leave independent roles unopened.

Scope: eight observed/twelve predicted annotation steps, detector-derived
image pixels, source-development evidence. Not metric/seconds, human gold,
physical safety, true3D or a foundation world model. Stage5C and SMC stay off.
