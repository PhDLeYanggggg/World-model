# Risk-Priority Repair: Benefit, Harm and Remaining Gaps

This descriptive decomposition uses the frozen readout. No policy, threshold, checkpoint, primary comparison or gate is selected from it.

## Predictive Result

Verdict: **advantage_but_screen_failed**. No deployment change.

| Comparison | Lost benefit pp | Avoided positive harm pp | Net gain/full floor pp |
|---|---:|---:|---:|
| risk_priority_matched_vs_uncapped_matched | -0.00448299 [-0.008759014, -0.001403789] | -0.0007382068 [-0.002127064, 0.0001528697] | 0.003744783 [0.001284861, 0.006803724] |
| uncapped_matched_vs_raw_matched | 0.02333829 [0.009191067, 0.04044589] | 0.005279427 [0.0009105499, 0.01236212] | -0.01805886 [-0.02945759, -0.007882786] |
| risk_priority_matched_vs_raw_matched | 0.0188553 [0.007468727, 0.03200743] | 0.00454122 [0.0009224092, 0.01031648] | -0.01431408 [-0.02324215, -0.006098974] |
| uncapped_joint_vs_raw_joint | 0.6480481 [0.3315415, 0.9972857] | 0.142722 [0.04104227, 0.251742] | -0.5053261 [-0.7850891, -0.2618754] |
| risk_priority_joint_vs_raw_joint | 0.5403279 [0.2479791, 0.8611699] | 0.1128179 [0.01153812, 0.2184758] | -0.42751 [-0.6824641, -0.2060724] |

For each view, error = floor error - benefit + positive harm. Net gain therefore equals harm reduction minus lost benefit. These columns share the full-floor denominator; neither they nor a changed intervention volume can replace the registered matched-query ADE or selected-risk results.

## Training Mechanism Check

The auxiliary cap was active on215,041/216,000 updates; mean alpha=0.03656038. Minimum pre-optimizer risk projection=0.6406072. Only 46/108 repaired heads have lower final direct-risk fitting loss than the control. Fitting monitors from the restored checkpoints agree with the CREATE training receipt. Bounding raw gradient norms is not a guarantee about AdamW steps, useful admissions or held risk.

The supervised monitor is the unweighted reporting sum, not the scalar objective implied by a time-varying detached cap. Compare the same monitor component between arms; do not rank these training rules by total loss.

| Arm | Held-development Brier | Signed-risk MSE | Signed bias |
|---|---:|---:|---:|
| risk_priority | 0.1993874 [0.1726962, 0.226943] | 0.004612871 [0.002597369, 0.007167478] | 0.006458902 [-0.0004102601, 0.01364686] |
| uncapped | 0.1581311 [0.1441006, 0.173253] | 0.004443021 [0.002675989, 0.006446756] | 0.008916193 [0.001876823, 0.01587688] |

The single changed factor supports a controlled comparison of this fixed cap, not a universal claim about gradient balancing. A lower composed risk loss, if present in a slice, need not improve utility-preserving joint allocation. Probability, conditional-moment bias and ranking need separate evidence; their causal mechanism is not identified by these scores alone.

## Structural Support and Generalization

At least19 dependent views are forced to abstain by unchanged anchors. Their undefined selected risk is retained. The every-view-defined-risk screen cannot be repaired by auxiliary gradient reweighting alone. Predicted feasible constraints are also not a certificate about realized harm. Independent calibration and confirmation remain missing; no eligibility or risk-definition change is authorized by this descriptive result.

Only12 already-opened development localities; repeated roles and the three forecast seeds are not independent samples. Intervals are nominal3,000-draw paired-locality intervals, not simultaneous or independent-confirmation intervals. Independent selection/calibration/confirmation remain closed. The original selected-risk primary remains incomplete; a full-floor denominator cannot replace it. Image-local detector silver; obs8/pred12 with raw-frame stride12. No metric, seconds-level, human-gold, physical-safety, true3D, foundation, calibration-certificate or submission-readiness claim. Stage5C and SMC remain disabled.
