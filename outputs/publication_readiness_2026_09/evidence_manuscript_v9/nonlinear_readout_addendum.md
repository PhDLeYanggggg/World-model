# Nonlinear Risk Readout Control: 5 November 2026

This evidence addendum does not replace or rebuild the main-paper PDF. All
previous failures and unavailable primary comparisons remain unchanged.

We fitted six two-output histogram-gradient-boosting readouts in three matched
contexts. One arm uses two frozen signed risk scores; the other adds the same
64 neural hidden features. Both preserve empirical query weights, fitting
sources, labels, missing-outcome bounds, proposal actions and the original
2% risk contract. Each output has 128 rounds. All six models freeze before
held-source readout, without early stopping or threshold selection.

Adding hidden features improves fitting MSE more than the score-only control:
40.87-64.59% reduction versus 11.26-39.10%. The corresponding fitting utility
contrast is +1.7243 percentage points. However, on seven exposed SOURCE units,
known ADE gain over the same floor is 4.3233% for the affine readout, 1.5845%
for score-only boosting and 1.5090% for latent-plus-score boosting. The latent-
minus-score contrast is -0.0755 points, with missing-outcome completion bounds
[-0.1807, +0.0296]. These bounds are not sampling confidence intervals.

All three registered latent-minus-affine completion-lower contrasts are
negative. Full original risk/support remains 0/17 recording views, with 13
known all-risk and 15 easy-risk violations. Two selected-easy views become
undefined. Reduced intervention discards useful actions without eliminating
harmful admissions. Thus neither nonlinear readout is promoted, and frozen
latent contribution is not established by this experiment.

The real resume pilot, first full independent fit and all three inference,
decision and metric blocks reproduce exactly. An initial evaluator metadata
lookup failure is retained and explicitly amended; weights and scientific
criteria were not changed. This is standard readout training, not new encoder
or trajectory training, a novel boosting method, or a safety theorem. No new
H, seed study or independent test was run. One-seed exposed development
results cannot support a final generalization claim.

This control weakens the explanation that only a linear risk-output bottleneck
prevents success. It does not prove that all causal representations lack risk
information. Further model work needs to address conditional risk on admitted
actions, rather than cite lower regression error or fewer interventions as
evidence of safe utility. The original 24-group primary remains NA; no identity
certificate is introduced.

Evidence: [complete report](../moving_nonlinear_readout_v1/README.md),
[action-flow diagnosis](../moving_nonlinear_readout_v1/failure_analysis.md),
[frozen protocol](../moving_nonlinear_readout_v1/protocol.md),
[exact replay](../moving_nonlinear_readout_v1/verification.json).

Native image-pixel obs8/pred12 development only. No verified seconds, metric,
human-gold, true 3D, foundation, deployment or submission-ready claim. Stage5C
and SMC remain disabled.
