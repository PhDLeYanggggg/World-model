# Source-Supported Start Information: Fixed Comparison

The previous 27-fit conditioning study failed protected forecasting. Earlier
proper-score start probes used only 81 ETH and 284 Hotel stationary windows,
with asymmetric transfer. SDD is now admitted as separate training-only source.
Hypothesis: source supervision under an explicit start classification objective
adds cross-site information that raw displacement losses did not capture.

Use the frozen past-only unit-frame v2 geometry, no images, identities or domain
labels. Stationary-query membership depends only on eight past coordinates.
Future annotation change is a separate supervised proxy, not human intention.
Source supervision uses complete twelve-step labels; incomplete stationary
queries remain counted as unscored, not relabeled as non-starts. Main/source
forecast populations and primary ADE are unchanged; this is a diagnostic head.

Fixed schedules: main-only, source-only, and mixed with one-half sample mass
per domain. Fixed families: logistic regression, ExtraTrees and a small MLP.
Three seeds, two main held-site directions. Source-only fits are shared across
both directions: 45 fresh models, 54 evaluation cells, not 54 independent fits.
MLP has 1,000 updates, checkpoint/resume. Trees checkpoint every32 estimators.
No hyperparameter, probability threshold, prior correction or held calibration
search. All fitted normalization uses only each schedule's training inputs and
weights. Source-only does not use target-domain normalization or training labels.

The shared evaluation reference is the opposite main training site's smoothed
prior, explicitly a stronger target-domain-informed comparator for source-only.
Also compare source/mixed against identical main-only model families. Require
consistent probability gains in both directions before arguing transferable
start information; even that cannot establish trajectory direction, risk,
joint intervention or deployment. Store every result and warning.

Primary probe score: Brier; AUROC/AUPRC/ECE/log loss and agent-balanced paired
intervals are secondary. Bootstrap averages seed losses, not probabilities.
Overlapping windows, contemporaneous agents and two exposed sites limit all
intervals; this is not independent confirmation. Zara has no stationary rows.

Preserve approved original SDD train40 only, stride12 raw frames. Do not equate
it to physical main-task time. No new dataset-role decision, sealed-role access,
evaluation-primary change, future input, central velocity or test-goal use.
No larger trajectory run follows automatically. Stage5C/SMC stay off.
