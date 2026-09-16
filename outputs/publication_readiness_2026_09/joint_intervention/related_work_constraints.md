# What Existing Work Already Establishes

Checked against original paper text on 2026-09-16. This is a focused, non-exhaustive comparison, not a novelty certificate. Scope: baseline-relative gain/harm, joint intervention and risk calibration. No model or dataset was sent to an external service.

## Joint Forecast Compatibility

[JFP: Joint Future Prediction with Interactive Multi-Agent Modeling for Autonomous Driving](https://arxiv.org/pdf/2212.08710), section 3.3, represents joint futures with unary and learned pairwise energies. Its appendix also tests hand-crafted overlap penalties as post-processing and during training; a heuristic can lower overlap while harming accuracy. Therefore, unary-plus-pair energy, graph-based trajectory compatibility and geometric post-processing cannot be claimed as new M3W contributions. The current MILP prototype is a baseline/control implementation. The research distinction to test is baseline-relative intervention with excess-risk supervision and calibration, rather than general joint mode selection. An accuracy-consistency tradeoff alone is not evidence of a better method.

[Joint Metrics Matter: A Better Standard for Trajectory Forecasting](https://arxiv.org/pdf/2305.06292), sections 1 and 3, distinguishes marginal best-of-K trajectory errors from errors of a single coherent joint sample and studies joint training losses. Our initial binary selector is deterministic: it must not borrow best-of-K claims, or pretend that average K=1 agent error alone captures interactions. Measure geometric composition effects explicitly while keeping the predictor fixed. The paper's standard 8-observation/12-prediction setting supports proposing a comparable main protocol, but adopting it still requires the project's protocol and source-timing checks.

## Risk Control

[Learn then Test](https://arxiv.org/html/2110.01052v5), sections 2.1-2.3, treats risk-constrained policy selection as multiple testing using an independent calibration sample. It describes Hoeffding-Bentkus tests for bounded losses and family-wise error control. Policy ordering learned from data needs appropriate splitting. Our simple simultaneous Hoeffding screen is a conservative engineering primitive, not the paper's complete algorithm or a new theorem. Learned policy candidates cannot be repeatedly changed after inspecting calibration failures while retaining the original family guarantee. Overlapping trajectory windows are not automatically independent calibration observations.

[Conformal Risk Control](https://arxiv.org/pdf/2208.02814), section 2.4, explicitly exhibits failure for non-monotone losses and discusses monotonic envelopes. A thresholded multi-agent intervention policy need not have monotone realized trajectory risk. Adding a "conformal" name or a validation quantile does not prove coverage, an easy-degradation bound, or safety under arbitrary domain shift. The finite-family screen is retained as an explicit alternative whose assumptions still need to be justified; no CRC guarantee is claimed for this implementation.

## Implications for the Paper

1. Proposed novelty must be evaluated against these mechanisms, not just against neural architectures without gating.
2. A same-forecast factorial comparison must isolate the value of realized excess-risk labels, coupling and calibration, at comparable intervention coverage or a declared risk budget.
3. Separate average raw ADE/FDE, normalized excess loss, clipped bounded risk, failure probability and geometric proximity. They are not interchangeable guarantees.
4. Six physical-scene groups are not enough evidence for a broadly useful scene-level safety guarantee. Report the support limitation; consider more independent recordings or narrower empirical claims subject to protocol approval.
5. No scientific claim can be based on the old duplicate/test-exposed external evaluations. New independent confirmation, matched public forecasters, three seeds and cluster-aware uncertainty remain missing.

Further literature work remains on selective regression/learning to defer, structured selective prediction, cross-fitting with clustered observations and forecasting under distribution shift. This focused check neither proves absence of an equivalent existing method nor establishes CVPR readiness.
