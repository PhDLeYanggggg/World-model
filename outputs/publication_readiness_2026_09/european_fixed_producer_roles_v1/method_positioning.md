# Method Positioning And What This Test Cannot Prove

Primary papers were reopened on2026-09-25. This is a focused positioning check,
not a systematic novelty review or a claim to have invented selective prediction.

## Cross-Fitted Supervision Is Established

Breiman's *Stacked Regressions* uses cross-validated first-level predictions to
learn a combination, avoiding the in-sample fit problem. It also discusses
instability from correlated predictors and constrained combinations. These are
established ideas, not M3W contributions.
[Original paper, Machine Learning24:49-64,1996, sections1-2](https://statistics.berkeley.edu/sites/default/files/tech-reports/367.pdf).

Our narrower experiment changes the producer chain used to supervise a cost
controller while holding the final trajectory candidate/floor fixed. It compares
source-disjoint matched forecasts with the older OOF chain. Any effect is jointly
affected by producer fitting sources/count, forecast quality, labels and feature
statistics. This does not identify producer identity alone, and a role split is
not itself a novel learning algorithm. Fixed weighted ridge is a useful simple
cost-head control, not an implementation of Breiman's trajectory combination.

## Predicted Risk Is Not Calibrated Risk

*Conformal Risk Control* gives an expected-risk result for its specified
calibration procedure under exchangeable, bounded, monotone loss functions;
Theorem1 also states continuity and endpoint conditions. Its distribution-shift
extensions have further assumptions.
[Original ICLR2024 paper, sections1-2 and4](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).

Our learned harm/reference ratio with a2% cutoff is not that algorithm. No
independent calibration role is opened here, and equal-locality bootstrap CIs do
not establish exchangeability or certify each scene. Selective net gain is not
automatically a monotone loss; positive harm and net harm are different targets.
Therefore this experiment can measure prediction/reliability failures, but
cannot inherit a conformal or physical-safety guarantee by using a risk label.

## Evidence Needed For The Project Claim

A contribution would need stable gain over strong same-predictor controls,
preserved easy behavior, useful scene-level joint decisions, appropriately
supported risk calibration, and untouched source-level confirmation. This small
cost-controller experiment does not retrain world dynamics, prove multimodal
scene understanding, or establish cross-dataset generalization. It tests one
specific failure mechanism before those larger claims can be considered.
