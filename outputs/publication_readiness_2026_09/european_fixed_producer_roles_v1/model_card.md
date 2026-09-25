# Fixed-Producer Matched Cost Controller

Status: promising development repair, not deployment promotion or independent
confirmation. The current deployment policy is unchanged.

## Model And Intended Use

The candidate is a pair of small Torch cost heads, not a newly trained trajectory
world model. A380-dimensional causal history/neighbor/rollout feature vector feeds
a width64 bounded utility head and a ranked hurdle-risk head. They have24,514 and
24,579 parameters. Fixed original forecasters and motion floors supply trajectories.

The utility outputs estimate benefit/harm versus the protected floor. The risk
outputs estimate a reference-error moment and positive-harm moment for the all or
easy training event. Predictions only intervene when the latest observed step
moves, predicted benefit exceeds harm, and predicted harm is no greater than2%
of the predicted reference moment. A causal geometric envelope bounds relevant
cost outputs; it is not a learned physical-validity or statistical-safety certificate.

## Training And Validation Boundary

Six ordered A/B/C rotations, seeds17/29/43 and two fixed event targets give36
groups. A supplies the frozen four-source producer/floor; B supervises the new
head; C is opened development readout. Both matched and OOF supervision arms are
newly fitted for2,000 final updates, batch256. A100-update pilot resumes inside
the budget. No C-selected checkpoint, threshold or event. No final-test tuning.

Matched and OOF controls have identical sampling counts, source weights, capacity,
budget and random seeds. Targets and feature/ranking normalizers differ because
the supervision producer differs. Some OOF fits repeat supervision; they are not
additional independent replications. A fixed-alpha weighted ridge control is also
included. Public reports retain every arm, rotation and adverse locality.

## Observed Evidence

The matched controller's worst positive-easy degradation is0.23767% versus CV,
below2% in all36 views. OOF and ridge controls reach7.85031% and8.23348%. Matched
all-ADE gains over the same four-source floor range0.06143% to2.56760%, with36
positive conditional locality intervals.

Against the stronger unchanged stopping-protected controller, gains range
-0.90074% to+1.64708%:12 positive and3 negative all-ADE intervals. Hard results
include5 negative intervals. This is not uniformly superior. Risk scores still
underpredict selected positive harm in122/144 dependent locality/views;66 exceed
the nominal2% predicted budget in realized outcomes. Net easy preservation and
positive-harm calibration are distinct. Neither is physical safety.

## Limitations And Reproduction

Only twelve opened EuropeanSquares localities and four per readout interval.
Detector tracks, overlapping windows, partial/missing labels and single-dataset
exposure limit the claim. No independent calibration, confirmation, new images,
new dynamics training, foundation or true3D result. Image pixels and raw-step8/12
only, not metric or seconds. Stage5C and SMC are disabled.

The native arm64 environment uses CPU4/inter-op1/workers0. Checkpoints and heartbeat
support resumption; all144 checkpoint prefixes and72 ridge score banks replay.
Source arrays/checkpoints remain private/local and are not redistributed. The
public configuration, scripts, loss traces, metrics, source hashes and operation
guide describe the exact reproducibility boundary.
