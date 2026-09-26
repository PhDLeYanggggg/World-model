# Easy-Membership Diagnostic Model Card

## Material Passport

Two source-development classifiers, not trajectory dynamics or a deployment
policy. Causal feature/producer lineage is cached_verified; new training and
probability predictions are fresh_run. All 288 fits and 576,000 updates are
complete; prediction freeze 269d7ec0 preceded held readout. Current checkpoints and
row-level probabilities remain private.

## Model and Target

Both models receive the same 383 causal features, including past context and
past-input producer forecast diagnostics. Linear has 384 parameters; width-64
GELU MLP has 24,641. Both produce one probability for the inherited label
0 < realized CV_error <= fitting_easy_cut. This includes easy rows without
alternative-model harm, unlike a positive-harm classifier. Exact-zero CV error
is excluded by the inherited label definition. Future errors are loss/eval
labels only; the prediction interface accepts only features and fitting
normalizers. No held label, central velocity or test endpoint goal is allowed.

Binary cross-entropy, equal-locality known-row sampling, no class reweighting,
2,000 updates, AdamW, fixed learning rate and seeds 17/29/43. Initial output
probabilities and sampled rows match between arms. Architecture capacity
differs, so their comparison is not a capacity-matched representation ablation.

## Interpretation

Brier skill and log loss compare against a training-prevalence constant.
A prospective sensitivity additionally uses training prevalence conditional
on positive disagreement. The reference D_easy/D_all score is a ranking proxy,
not an event probability. Neither AUROC nor ECE alone certifies safe switching.
A positive membership diagnostic does not establish conditional harm magnitude,
policy improvement, independent calibration or a deployable world model.

No new policy, trajectory, Stage5C or SMC execution. Eight observed/twelve
predicted annotation steps, image pixels and detector-derived labels, not
verified metric/seconds, human gold, true 3D or foundation capability.
