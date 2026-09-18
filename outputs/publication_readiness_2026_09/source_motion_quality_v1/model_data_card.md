# Probability Probe Model And Data Card

Purpose: diagnose whether fixed past geometry and box shape carry transferable
motion-occurrence information. This is not a released predictor, dynamics model,
physical safety guarantee or standalone contribution.

Data: unchanged 15,430 stationary-history SDD queries, 29 recordings, 545
recording-scoped agents, four previously explored sites. Eight observed and
twelve future annotation steps with stride 12. Observed centers are constant.
Overlapping windows are dependent, and fold training sets overlap. Bookstore
and main/sealed roles are not fitted or forecast. Shared infrastructure loads
identity-verified corpus arrays; exclusion does not mean zero file access.

Inputs: 480 fold-normalized observed-unit geometry/rotation columns, optionally
38 scale/translation-invariant past-box width, height, differences and aspect
features. No future labels, future boxes, source generated/occlusion flags,
remaining track length or query identity are predictor features.

Labels: any future coordinate change and half-observed-box maximum excursion.
Future labels are legitimate training supervision and held scoring only. They
do not certify human intent or actual physical movement. All rows are retained.

Model: ExtraTreesClassifier, 128 trees, depth 8, minimum leaf 64,
max_features 0.5, no class weighting. Seeds 17/29/43, four held sites, two tasks,
two inputs: 48 terminal models. No selection/calibration threshold is fitted.
Four fitting threads; serial tree reduction for exact prediction replay.

Reference: training-complement prevalence probability. Scores: Brier and lift,
AUROC, AUPRC, ECE and ten-bin calibration. Added-feature contrasts share 2,000
four-site bootstrap draws. Negative mean Brier lifts in all four variants.
Feature-count changes affect forest random subspaces, so the matched procedure
does not isolate a universal causal effect of box information.

Observation boundary: supplied offline annotations, not certified sensor-as-of.
15,316 histories use generated annotations with a later-than-query control.
No direct future-target feature path is detected by the scoped tests; original
annotation acquisition is a different boundary. No seconds/metric calibration.

Reuse: private models and row-level probabilities stay outside Git. Public code,
config and aggregates require matching locally acquired data and prior receipts
to reproduce. Do not deploy these classifiers as a trajectory switch or report
their oracle labels as inference inputs. Stage5C and SMC remain disabled.
