# Nested Calibration: Partial Protection, No Neural Advantage

Status: completed and reproduced; not a deployment or submission-ready result.

## What Was Run

I tested whether source-held calibration can repair the selection-dependent
harm underestimation found in the symmetric-risk study. The fitting chain now
excludes calibration and outer localities all the way through the forecast
producers, cost labels, preprocessing and score heads.

Fresh training: 18 two-locality Transformer producers, 72,000 updates, followed
by 54 gain/harm or event-risk heads, 108,000 updates. Nine frozen four-locality
final producers are cached_verified. Every new predictor and head was replayed
from its checkpoint. Both predefined role rotations and all three seeds remain
in the report. No outer result selected a model, threshold or checkpoint.

Population: 318,969 past-eligible targets from 12 opened source localities;
311,922 have ADE labels, 240,269 have final-step labels, 193,705 have all future
steps, and 7,047 remain future-unknown. These are overlapping detector-track
windows, not 318,969 independent examples. Each assignment uses four fitting,
four calibration and four outer localities. Reserved data roles stay closed.

## Main Result

| Candidate | Calibration | Full observed safety | ADE gain vs CV range (%) |
|---|---|---:|---:|
| Neural | None | 2/12 | -0.1449 to 1.5213 |
| Neural | Population rescale | 5/12 | -0.0871 to 1.0287 |
| Neural | Selected-risk grid | 5/12 | 0.1467 to 0.5308 |
| Damping 0.97 | None | 10/12 | 1.2151 to 3.0362 |
| Damping 0.97 | Population rescale | 12/12 | 0.9654 to 2.4839 |
| Damping 0.97 | Selected-risk grid | 11/12 | 1.1937 to 2.7938 |

Each row retains three seeds, two rotations and two event targets. Full observed
safety requires worst-locality positive-easy degradation <=2% and no added
error on the observed zero-CV cases. These correlated views are not independent
replications or a statistical safety certificate. Positive harm budgets and net
easy degradation are distinct quantities.

**Every one of the 36 direct neural-versus-matched-damping ADE comparisons is
negative.** The difference ranges from -3.0763% to -0.6859%; 34 conditional
intervals are strictly negative, two cross zero, and none supports neural
superiority. All 36 hard and all 36 easy point estimates also favor damping;
31 hard and 27 easy intervals are strictly negative.

Calibration therefore improves some protection but does not establish the
proposed neural dynamics advantage. Against its matched uncalibrated neural
control, population rescaling improves only 1/12 point estimates and the grid
improves 2/12; neither has a strictly positive improvement interval. Protection
usually costs utility. A positive gain over CV alone is not enough.

## Why Calibration Is Not Certification

The selected-risk grid satisfies its positive-harm restrictions in all 36/36
neural calibration assignments by construction, but only 27/36 outer
assignments satisfy the same empirical restrictions. Damping also falls from
36/36 to 27/36 under this stricter positive-harm check, even though its net easy
preservation is better. Four calibration localities do not establish transport
to another locality. The two rotations share the same 12 opened sources.

One frozen neural rule produces 6.4081% positive easy harm on outer locality020;
the corresponding policy's worst-locality net easy degradation is 5.8425%.
Several other rules harm observed zero-CV cases. Those outcomes are retained,
not filtered out using future validity. Two neural calibration maps choose full
abstention; neither is evidence of positive transfer.

The raw neural candidates achieve 2.4167--3.0561% ADE gain over CV across
seed/rotation pairs, but all their conditional intervals cross zero. Fixed raw
damping achieves 3.9755%, CI [0.9038%, 5.8037%]. This points to candidate
quality and source variability as well as scoring error; it does not prove
that better future labels or new features cannot improve the neural model.

## Verification and Decision

18 producer replays, 54 head replays, 216 independently reconstructed decision
banks, nine matched sampler groups and complete metric replay pass. All 196
tests in 28 scoped files pass; the full legacy suite was not run. Each reported
interval uses 3,000 locality resamples conditional on the fitted source models.
They are not multiplicity-adjusted independent discoveries.

No deployment promotion. No joint-controller benefit is established by this
pointwise experiment. Historical Stage35/37 scores are not recertified.
No Stage5C execution, SMC, metric, seconds, human-gold, physical-safety, true3D
or foundation claim. The data remain released detector tracks in image pixels,
obs8/pred12 with raw stride12, not historical raw-frame t50.

Next: use the saved models to separate inner/final-producer shift from source
shift and opportunity ranking failure before registering another loss change.
Do not continue tuning the frozen grid on these outer outcomes, and do not open
reserved confirmation data to rescue this negative source-development result.

[All results](results.md), [calibration transport](calibration_transport.md),
[figure](calibration_comparison.svg), [gates](gates.md),
[failure analysis](failure_analysis.md), [execution](execution_notes.md).
