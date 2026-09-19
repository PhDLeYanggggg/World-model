# Motion-Information Probability Probe

Registered while the fixed motion trajectory experiment is running, before
reading its aggregate result. This is a complementary source-only diagnostic,
not a replacement metric, new primary task or deployment gate.

Question: does the measured past motion add information about subsequent
annotation change, even if the trajectory readout stays near the stationary CV
solution? A positive probability probe would not prove motion direction or ADE
improvement. A negative linear probe would not prove absence of all information.

Use the same 15,430 source windows, four excluded-site folds, guarded past
geometry and train-normalized flow token store. Quality-only versus quality+motion
is the fixed contrast. All rows retained. No main/bookstore/external readout.

Fit uniform-row logistic regression, C=1, LBFGS, max_iter=2000, tol=1e-8,
no class balancing, no hyperparameter/threshold selection. Convex solver gives
one deterministic fit per site/arm/target, not three duplicated seeds.

Two fixed labels: any nonzero supplied future displacement, and maximum supplied
future excursion >10 annotation pixels. The latter threshold was already used
by the earlier motion-quality audit; it is not a physical movement definition.
Both use future labels only in fitting/evaluation. Sixteen total models.

Report Brier, log loss, AUROC, AUPRC, positive prevalence and a train-prevalence
constant predictor. Primary diagnostic contrast is paired Brier reduction;
all other measures retained. Two thousand conditional resamples of four explored
sites, not independent rows or confirmation. Single-class site metrics are
undefined, not zero. Save coefficients, identities, iteration counts, predictions
and per-fit receipts; completed fits resume without refitting. Fail visibly if
solver convergence is not attained. No trajectory gate changes from this probe.

No deployment, formal calibration, sensor-as-of, metric, seconds, true3D or
foundation claims. Stage5C and SMC remain off.
