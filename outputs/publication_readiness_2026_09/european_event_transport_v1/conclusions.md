# Common-Event Labels Do Not Repair Residual Transport

## Verdict

The single-factor common-event repair fails its mechanism, primary and
tail/coverage gates. Keep the original risk estimator and deployment unchanged.
This is a completed source-development negative result, not a trajectory gain,
independent confirmation or submission-ready method.

## What Was Run

- fresh_run:864 fixed global/context ridge fits,144 views,36 readout groups,
  all three seeds and both input pairs. No folds dropped or thresholds tuned.
- cached_verified:432 inner neural heads, frozen original outer estimators,
  trajectory producers, old inner-event probes and prior three-locality repair.
- not_run:new neural/trajectory training, intervention utility, independent
  selection/calibration/confirmation, Stage5C and SMC.

Registration08f94699 preceded fitting; prediction freeze328856c8 was committed
before fresh outer readout. All old outer target hashes match. The new common
event is defined using three meta-fitting localities only; it is never used
to refit the two-locality teacher or placed in inference features. OOF refers
to teacher prediction provenance, not independence of meta-supervised labels.

## Main Evidence

Positive means lower easy-harm MSE on known positive-disagreement rows.
Ranges below contain six point estimates, not one confidence interval.

| Common-event OOF context compared with | Inputs | Positive / negative / overlapping 95% intervals | Point range (%) |
|---|---|---:|---:|
| Original estimator | Full | 0 / 1 / 5 | -3.9629 to +0.0504 |
| Old inner-event OOF context | Full | 0 / 1 / 5 | -3.9816 to +1.8992 |
| Prior three-locality context | Full | 0 / 4 / 2 | -2.7362 to -0.2278 |
| Common-event global bias | Full | 1 / 1 / 4 | -2.2671 to +1.0183 |
| Common-event cyclic next | Full | 0 / 2 / 4 | -2.7638 to -0.4733 |
| Common-event cyclic previous | Full | 0 / 1 / 5 | -3.9236 to -0.3495 |
| Original estimator | Motion-only | 0 / 0 / 6 | -19.9452 to -1.6391 |

The coverage comparison with old OOF has one negative interval; top10 harm
capture has two. Secondary positive ranking or cyclic-control contrasts do
not replace failed primary comparisons. Every registered arm is retained in
`results.md` and the all-contrast figure.

Each contrast averages three seeds within locality, then uses3000 paired
resamples of four localities. Six overlapping assignments are not independent
studies. These intervals are not multiplicity-adjusted and cannot establish
safety, noninferiority or equivalence. Motion-only point estimates are all
negative even though its original-comparator intervals overlap zero.

## What The Factor Check Establishes

All864 fits satisfy the exact preclip ridge identity: the difference between
old and common-event shifts equals the projection of their supervised label
difference. Maximum coefficient identity error is8.1411e-15. This confirms
the implemented factor isolation; it does not establish a causal explanation
of the original failure. Clipping is explicitly accounted for and is nonlinear.

Simply making the residual response use the outer-fitting event is not
sufficient. Inner predictions still describe their original two-locality event,
and their errors need not transfer to a stronger three-locality outer head.
The seven-feature additive residual is not reliably helpful in this setting.
Neither a larger model nor more threshold search follows from this result.

Summed per-view fitting/inference time is80.064750seconds, excluding surrounding
source loading and preflight. This is not total wall time or neural training.
Full replay status belongs in `verification.json`, not inferred from process
completion. Independent roles remain unopened; the research goal remains active.

Scope:eight observed/twelve predicted annotation steps, detector image pixels.
No verified meters, seconds, human gold, physical safety, true3D or foundation
claim. No new deployment, Stage5C execution or SMC.
