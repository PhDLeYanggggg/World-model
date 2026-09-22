# Standard Cost-Forest Comparator: Fixed Before Training

## Material Passport

2026-09-22. Source-only experiment, new forest fits not yet run at registration.
Based on the completed temporal intervention and fit/held support diagnosis.
This is standard-model comparison, not a new tree method or a one-factor loss
experiment. Author-provided eight-observation/twelve-prediction task; annotation
steps at SDD stride12, annotation pixels, no seconds or metric interpretation.
All four sites have informed development. No independent calibration, final-test
claim, new deployment, Stage5C or SMC.

## Question and Fixed Scope

Can an ordinary cost-sensitive ExtraTrees regressor learn useful conditional
benefit/harm decisions on the same causal features where the neural cost head
underestimates harm on admitted held-scene rows? Success here would make the
neural risk-head claim weaker unless it later beats this strong comparator.
A negative result does not prove feature insufficiency or irreducible ambiguity.

Use coupa/deathCircle/gates/hyang, seeds17/29/43, two existing causal candidates
(temporal ramp and displacement-matched uniform). Freeze the existing neural
forecasts and complete fitting support. Each held-site head uses the three other
sites; upstream fitting-row producers exclude both that row's site and the outer
site. No new roles assigned. Existing train-only feature standardization, easy/
hard thresholds and exact past-stop rule remain unchanged. Future labels are
supervision/readout only. No future-availability feature or new endpoint goals.

Fit24 joint two-output forests,128trees, depth16, minimum leaf64 unique rows,
max_features1/3, bootstrapfalse, squared_error. Seeds fixed above, fit4threads,
predict1thread for stable sums; sklearn1.8.0, native arm64, no worker processes.
Save every16trees and record PID/heartbeat. A16tree timing/resume pilot does not
evaluate held outcomes or select settings. Full budget is128trees per endpoint.
No hyperparameter, threshold, candidate, seed, subgroup or role search.

Targets are benefit/D and harm/D, where D is past/forecast-only mean native
disagreement. Cost simplex leaf averages are scaled by D at inference. D=0 gets
exactly zero predicted cost. Unknown fitting targets have zero training mass;
observed partial/unknown outcomes remain in the inference population/readout.
Weights are the actual frozen neural sampler counts times its fixed region loss
weight times D/train_cost_scale, normalized by one common positive factor. This
matches row exposure emphasis, not the logarithmic objective, optimizer, model
capacity or compute. The forest sees positive-weight unique rows, not duplicated
draws for leaf-size counting. Record effective rows, nodes, fit time and disk use.

## Fixed Decision and Readout

Primary: ramp forest versus cached verified ramp neural under the same strict
benefit>harm and harm<=0.1benefit gate, D>0 and nonzero last past displacement.
Uniform forest is secondary; never promote a secondary winner after readout.
Within each held scene/seed/arm, neural net-gain ranking is also evaluated at
exactly the forest switch count, eligible pool fixed, ties by row ID. Equal
count is not equal intervention distance. No outcome-dependent ranking.

Report native ADE/FDE, equal physical-site percent gain versus causal CV, hard,
positive-error easy, exact-zero-CV harm, tails, worst site/seed, known/partial/
unknown support, full-grid gain bounds, switch count and selected disagreement.
Use3000 paired physical-site bootstrap resamples, seed38113. Four development
sites and three repeated seeds are not thousands of independent observations.
Compare conditional risk of both estimators on both fixed selected regions.
Report discrete smoothness and existing past-context proximity with unsupported
edges retained; neither is a physical safety certificate or joint optimization.

The primary empirical conjunction requires paired ADE difference CI lower>0,
positive CV gain each seed, easy degradation<=2% aggregate and each site/seed,
and zero harmed complete exact-zero-CV cases. It cannot license deployment or
independent confirmation. Keep all negative results and the primary ADE rule.
Raw-frame t+50 supplement/external confirmation/full forecaster retraining are
not_run in this comparison; no historical Stage26/37 recertification.

## Reproduction and Attribution

Entry: `.venv-pytorch/bin/python scripts/run_m3w_forest_cost.py`.
Run `--audit-only`; train pilot `--view coupa_seed17 --arm ramp --stop-at 16`;
then full `--resume`, `--evaluate`, `--verify`. Training files/checkpoints and
row-level decisions stay private. Config/code/source hashes bind all endpoints.
Unit tests exercise real fit/resume, zero-disagreement, unknown labels, weighting,
simplex bounds and equal-count choice. Separate arithmetic verification follows
readout; shared implementation/data does not constitute independent research.

ExtraTrees is an established comparator: Geurts, Ernst and Wehenkel (2006),
[Extremely randomized trees](https://doi.org/10.1007/s10994-006-6226-1).
The author preprint's algorithm section2.1 and installed-version
[scikit-learn1.8 API](https://scikit-learn.org/1.8/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html)
were checked for random splits, whole-sample fitting, multioutput prediction,
sample weights and warm-start support. No claim of a full literature review or
novelty from swapping the estimator.
