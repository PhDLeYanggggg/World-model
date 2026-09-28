# Fixed Occurrence: Benefit and Harm

Descriptive decomposition of the frozen readout; no model, threshold, eligibility or gate changes. For each view, error = floor error - benefit + positive harm. Every column below uses the same full-floor denominator, which is not the selected-risk denominator or primary ADE denominator.

| Comparison | Lost benefit pp | Avoided positive harm pp | Net gain/full floor pp |
|---|---:|---:|---:|
| fixed_matched_vs_trainable_matched | 0.02145914 [0.009063314, 0.03519416] | 0.00529482 [0.001794977, 0.009193507] | -0.01616432 [-0.02857015, -0.005905077] |
| trainable_matched_vs_raw_matched | 0.04980784 [-0.005258046, 0.09616399] | 0.01296979 [0.003137462, 0.02336186] | -0.03683805 [-0.07855897, 0.01021586] |
| fixed_matched_vs_raw_matched | 0.07126698 [0.01298541, 0.1269848] | 0.01826461 [0.00540809, 0.03220034] | -0.05300237 [-0.1031922, -0.003979079] |
| trainable_joint_vs_raw_joint | 0.2903527 [0.08949238, 0.5131436] | 0.06587688 [0.01933925, 0.1173019] | -0.2244758 [-0.4119356, -0.05938598] |
| fixed_joint_vs_raw_joint | 0.2721374 [0.0712468, 0.4912918] | 0.05865468 [0.005242195, 0.1148872] | -0.2134827 [-0.3917253, -0.05369715] |

Positive lost benefit is unfavorable; positive avoided harm is favorable. Their difference equals the net change. Matching each query count rules out intervention volume alone as an explanation. These quantities do not identify the causal explanation for changed ranking or domain shift.

The unchanged raw anchor forces at least16 views to abstain. All unsupported and risk-violating views remain in the evaluation; undefined ratios are not zero. The null solver-certificate repair is an engineering fallback, not learned risk calibration. See results.md for all policies, quality scores and the unchanged gates.

Only12 already-opened development localities; repeated roles and the three forecast seeds are not independent samples. Intervals are nominal3,000-draw paired-locality intervals, not simultaneous or independent-confirmation intervals. Independent selection/calibration/confirmation remain closed. The original selected-risk primary remains incomplete; a full-floor denominator cannot replace it. Image-local detector silver; obs8/pred12 with raw-frame stride12. No metric, seconds-level, human-gold, physical-safety, true3D, foundation, calibration-certificate or submission-readiness claim. Stage5C and SMC remain disabled.
