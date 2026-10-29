# Shared-Reference Action-Cost Head

Research-only54-head ablation,not deployable. Same25,157parameters and frozen
source trajectory bank as the unshared parent. Candidate inputs are causal
motion/rollout features plus action identity; a shared causal floor token
predicts R/ER. B/H/EH remain candidate-specific. Five nonnegative moments obey
the same envelope bounds; R/ER invariance is not a safety guarantee.

All heads use physical-fitting-only normalization and original sampled rows.
Future observations appear only in fitting labels/evaluation. Unknown labels
never enter regression loss; unknown eligible inputs remain in readout.
No central velocity,test endpoint goals or test statistics enter inference.

Native Torch CPU4,interop1,workers0. Checkpoint100updates,heartbeat50updates,
10GiBdiskreserve and12hhealthy-runtime boundary. All54fits and144group
readouts complete,hash-bound and exactly replayed. Weights remain private.

Known-label mixed ADE gain0.542017%,but paired neural increment is not
positive and original required-risk support is only21/186views. Same-count
comparisons are utility-only; undefined risk is not a pass. SOURCE inner-held
records were already exposed. No independent confirmation,whole-pipeline OOF,
metric/seconds,true3D,foundation,Stage5C orSMC claim. Deployment floor unchanged.
Detailed results and limitations:[readout](readout.md).
