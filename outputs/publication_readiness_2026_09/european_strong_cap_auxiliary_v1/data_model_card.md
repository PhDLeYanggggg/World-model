# Strong-Base Cap Auxiliary: Data and Model Card

## Purpose
Test whether a producer-relative cap-event auxiliary improves expected
forecasting-cost prediction without changing the strong estimator's base.
This is a source-development risk-estimation experiment, not a new trajectory
model, calibration procedure or deployment policy.

## Data and Roles
Reuse checksum-verified source windows and producer forecasts. Each view has
three fitting localities and one held source locality; forecast producers
exclude risk-model localities. Fitting event teachers additionally exclude
the row's own locality. There are six overlapping producer/controller
assignments, three seeds and full/motion forecast families, totaling144views.
The source pool is development-exposed. None of these folds becomes an
independent test by being recomputed. Independent selection, reserved risk
calibration and confirmation remain unopened.

Observation8/prediction12 native annotation steps, detector-image pixels.
No verified time/metric/physical-safety, human-gold, true3D or foundation claim.
No test-endpoint goals or central-velocity inputs. Future errors and event
labels appear only in supervision or later evaluation, not inference.

## Model and Training
The original383 native causal inputs, fitting-only normalization, GELU64
network, four-output cost objective and all-known site-balanced sampler are
preserved. Event labels are valid only on known positive-disagreement rows.
Their mask does not remove zero-disagreement rows from cost training.
Unknown labels are never drawn. Event probability never multiplies costs.

Three arms share initial weights, samples and fixed budget: cost only,
true-event auxiliary, within-locality-shuffled auxiliary. The original
membership intercept is retained solely for exact control reconstruction;
it is not initially calibrated cap probability. All144 no-auxiliary controls
must reproduce the frozen original within the registered numerical tolerance.
Actual bitwise equality is reported separately. Training learns all four
cost outputs; held comparisons freeze reference D/D_E and compare H/H_E.

## Limitations and Use
Two-locality fitting teachers and three-locality held teachers differ.
Rare-event support, distribution shift and magnitude estimation may remain
limiting even if classification improves. Full/motion comparisons also change
forecasts and outcome populations, so are not isolated input ablations.
Four localities per assignment limit uncertainty resolution. Three-seed
averages and3000 locality resamples do not remove prior exposure or assignment
overlap. Counts of per-view gains are descriptive, not independent trials.

No selection rule or policy threshold changes. Keep prior deployment unless a
separately authorized and independently validated policy study supports a
replacement. Stage5C and SMC remain off.
