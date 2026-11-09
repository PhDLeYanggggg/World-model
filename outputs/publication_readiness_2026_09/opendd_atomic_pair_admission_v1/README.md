# OpenDD: Whole-Pair Admission Does Not Survive Calibration

The exact-slot audit showed that rejecting edits one at a time can
remove the opportunity to make a useful joint choice. I trained a
model to score the two edits as a single decision instead. I then
checked whether the selected pair could be released on a location
that had not been used for fitting.

## Comparison

For each of seven held OpenDD locations, the next location in a fixed
cycle was used for calibration and the other five were eligible for
training. I fitted all 21 location-seed models (17, 29, 43). Training
used a fixed sample of up to 24 legal pairs per query; locations
without eligible pairs contributed no training examples. Each fit
had 86 to 132 strictly positive training slates. The labels require
local conservative gains over CV and OLS4, easy support and the three
original 2% checks. The additive head sums independent action scores.
The joint head adds a symmetric term based on past relative motion
and the proposed edit geometry.

All three roles use the same analytic paths: OLS4, the registered
motion variant, tapered turn and decaying acceleration. Two positive
past hindcasts nominate the peak slots before either head sees them.
This clean nomination is different from the old learned slot release:
edit counts cannot be compared directly with the preceding capacity
audit. Within this study, additive and joint have identical actors
available, candidate paths, query membership, edit counts and peak
offsets. The joint probability controls release for both arms.
No singles are released in this experiment. All features use the
eight observed positions; future twelve-step data are labels only.

For the gated evaluation, I picked the predeclared score threshold
that retained the most pairs while both arms passed all recording
checks on calibration, including conservative improvement over both
motion baselines. A fold with no nonempty feasible threshold returned
the exact CV prediction. This is development calibration with one
location per fold, not a formal risk guarantee.

## Results

| Policy | Edits | Risk passes / edited seed views | Known ADE | Available FDE | Conservative CV gain (pp) |
|---|---:|---:|---:|---:|---:|
| CV | 0 | Not an edited policy | 0.54726 | 1.25329 | 0 |
| OLS4 | Unmatched | Unmatched | 0.60906 | 1.34583 | −10.6805 |
| Forced additive | 966 | 8/159 | 0.54758 | 1.25391 | −0.0853 |
| Forced joint | 966 | 9/159 | 0.54784 | 1.25445 | −0.1005 |
| Calibrated additive / joint | 0 | Not estimable | 0.54726 | 1.25329 | 0 |

No calibration fold could release a nonempty policy. The forced
comparison was evaluated on 606 recording-seed views (202
recordings), including 159 edited views. The full-grid pair gain
for joint over additive is **−0.001910** native units, with a
3,000-draw paired location bootstrap interval **[−0.002719,
−0.001109]** over four estimable locations. One recording had a
qualified positive pair gain, but the average went in the opposite
direction. Forced joint exceeded 2% signed easy degradation in 52
views, with an undefined edited-easy reference in 48. Its CV gain
interval was [−0.2030, −0.0179] pp. The empty calibration comparison
is not an accuracy gain or a passed risk check.

## Why It Failed

After the scoring run, I audited calibration locations only. Across
**483 distinct nonempty score prefixes** summed over folds, **none**
met the recording criteria. This rules out a coarse threshold grid
as the explanation for this model. There were **174 calibration
query-seed views** where exhaustive local enumeration found a
strictly feasible pair, corresponding to **58 unique windows**.
The model ranked a locally feasible pair first in just **15/174**.
Local feasibility is not a guarantee at recording level. The learned
ordering also failed to separate successful and unsuccessful first
choices reliably: held-fold AUROC was 0.494 on the four folds with
both classes. The exact-prefix audit did not change any prediction,
threshold, split or reported outcome.

The next experiment should train the model to rank **feasible versus
infeasible slates within the same query**, mining difficult negatives
with matching radii. Its first benchmark should be capture of the
feasible calibration menu, not another threshold search. Only revisit
deployment calibration if that ranking improves. The current result
is a negative development ablation, not new submission-ready or
independent evidence. Stage5C and SMC stay off.

## Reproduce

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_atomic_pair_admission.py freeze
.venv-pytorch/bin/python scripts/run_m3w_opendd_atomic_pair_admission.py score
.venv-pytorch/bin/python scripts/audit_m3w_atomic_pair_calibration.py
```

The real arm64 fit and forecast pass took 18.07 seconds, under 507 MB
maximum RSS, with four Torch threads and zero loader workers.
Scoring took 3.48 seconds. Fits save model, optimizer and sampler
state every 50 steps and can resume through the same freeze command.
Sixteen focused tests passed. No CREATE job or reserved outcome access.
[Registration](registration.json), [metrics](summary.json),
[calibration-only audit](calibration_audit.json).
