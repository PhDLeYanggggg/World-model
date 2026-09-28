# Risk-Priority Learning: Registered Development Readout

Source: fresh_run paired Torch training, causal actions and development evaluation; upstream inputs/forecasters/floor/utility are cached_verified. All108 action groups and the complete numerical readout have exact replays. Independent confirmation: not_run.

The only training change caps the auxiliary-gradient norm at half the direct-risk-gradient norm. Architecture, initialization, sampling, AdamW, 2,000 updates/head, node budget and the2% risk budget are fixed. No threshold search or held-outcome model selection was performed. A pre-optimizer Euclidean gradient bound is not an AdamW-descent or safety guarantee.

**Verdict: advantage_but_screen_failed. No deployment change.**

The frozen raw/uncapped anchors already force at least19 dependent views to abstain under common-query-count matching. This was checked before the new readout; the defined-selected-risk gate cannot pass merely by changing the auxiliary gradient. These views and the registered gates are retained, not excluded after seeing results.

## Registered Contrasts

Primary: risk_priority_matched versus uncapped_matched. Each query has the same intervention count. Positive ADE gain and positive harm reduction are favorable. The all-floor harm column is only a diagnostic.

| Contrast | ADE gain % [nominal95% CI] | All-floor harm reduction pp | Selected-risk reduction pp |
|---|---:|---:|---:|
| risk_priority_joint_vs_raw_joint | -0.4472097 [-0.7210563, -0.2121746] | 0.1128179 [0.01153812, 0.2184758] | undefined |
| risk_priority_matched_vs_raw_matched | -0.01435295 [-0.02330288, -0.006110538] | 0.00454122 [0.0009224092, 0.01031648] | undefined |
| risk_priority_matched_vs_uncapped_matched | 0.00374642 [0.001285148, 0.006807738] | -0.0007382068 [-0.002127064, 0.0001528697] | undefined |
| uncapped_joint_vs_raw_joint | -0.526243 [-0.8253454, -0.2683482] | 0.142722 [0.04104227, 0.251742] | undefined |
| uncapped_matched_vs_raw_matched | -0.01810557 [-0.02954411, -0.00789567] | 0.005279427 [0.0009105499, 0.01236212] | undefined |

## Every Policy, Including Failures

| Policy | ADE gain/floor % | Hard gain/floor % | Intervention fraction | Risk-violating views | Undefined-risk views | Worst easy gain/CV % |
|---|---:|---:|---:|---:|---:|---:|
| floor | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 | 216 | 0.1214318 |
| common_anchor | 0.00279993 [0.0006956439, 0.00581501] | -0.0002229665 [-0.0006735095, 1.044148e-05] | 0.01017483 [0.006137437, 0.01443706] | 35 | 25 | 0.1646446 |
| raw_independent | 0.2688088 [0.1490469, 0.4050422] | 0.2543889 [0.1125564, 0.4187488] | 0.07798505 [0.05843923, 0.09724543] | 65 | 16 | 0.1766465 |
| raw_joint | 0.507865 [0.263107, 0.7901987] | 0.557981 [0.273242, 0.8755339] | 0.07798505 [0.05843923, 0.09724543] | 84 | 16 | 0.1766465 |
| raw_matched | 0.0229192 [0.01174609, 0.03528558] | 0.02137717 [0.00694832, 0.03862241] | 0.01017483 [0.006137437, 0.01443706] | 63 | 25 | 0.1646446 |
| uncapped_independent | 0.0003752045 [-0.01002544, 0.007246035] | 0.02129374 [0.0007482186, 0.0554662] | 0.01318076 [0.008361987, 0.01809935] | 56 | 6 | -1.289741 |
| uncapped_joint | 0.002538874 [-0.008656407, 0.01020307] | 0.0243302 [0.003680494, 0.05696087] | 0.01318076 [0.008361987, 0.01809935] | 73 | 6 | -1.289741 |
| uncapped_matched | 0.004860343 [0.001869181, 0.008409432] | 0.002594067 [0.0002824519, 0.00531562] | 0.01017483 [0.006137437, 0.01443706] | 51 | 25 | 0.1646446 |
| risk_priority_independent | 0.0407933 [0.01832721, 0.07024357] | 0.06263228 [0.01829257, 0.1244752] | 0.03707626 [0.02803326, 0.04596576] | 97 | 0 | -0.7195134 |
| risk_priority_joint | 0.08035493 [0.03618562, 0.1350199] | 0.1101643 [0.04785658, 0.1880651] | 0.03707626 [0.02803326, 0.04596576] | 120 | 0 | -0.7195134 |
| risk_priority_matched | 0.008605126 [0.003969607, 0.0143415] | 0.005344257 [0.0008533451, 0.01147875] | 0.01017483 [0.006137437, 0.01443706] | 54 | 25 | 0.1646446 |

An undefined selected-risk ratio is not zero risk and cannot pass the risk gate.

## Risk Prediction Quality

| Arm | Brier | Signed-risk MSE | Signed bias | Conditional reference MSE | Conditional harm MSE |
|---|---:|---:|---:|---:|---:|
| risk_priority | 0.1993874 [0.1726962, 0.226943] | 0.004612871 [0.002597369, 0.007167478] | 0.006458902 [-0.0004102601, 0.01364686] | 0.02492255 [0.018027, 0.0319008] | 0.01573053 [0.008553795, 0.02403468] |
| uncapped | 0.1581311 [0.1441006, 0.173253] | 0.004443021 [0.002675989, 0.006446756] | 0.008916193 [0.001876823, 0.01587688] | 0.005211583 [0.00437707, 0.006064022] | 0.01015169 [0.006000445, 0.01473215] |

## Gates

- matched_count: True.
- matched_ADE_advantage: True.
- matched_all_reference_harm_reduction: False.
- every_view_defined_risk_within_2percent: False.
- every_view_easy_preserved: True.
- no_zero_CV_harm: True.
- exploratory_screen_pass: False.
- independent_confirmation: False.
- formal_primary_replaced: False.
- calibration_certificate: False.
- deployment_changed: False.
- stage5c_executed: False.
- smc_enabled: False.

## Scope

Only12 already-opened development localities; repeated roles and the three forecast seeds are not independent samples. Intervals are nominal3,000-draw paired-locality intervals, not simultaneous or independent-confirmation intervals. Independent selection/calibration/confirmation remain closed. The original selected-risk primary remains incomplete; a full-floor denominator cannot replace it. Image-local detector silver; obs8/pred12 with raw-frame stride12. No metric, seconds-level, human-gold, physical-safety, true3D, foundation, calibration-certificate or submission-readiness claim. Stage5C and SMC remain disabled.
