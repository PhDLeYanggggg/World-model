# Easy Occurrence / Conditional-Risk Experiment

Registration preceded fitting. Forecasts, floor, utility and all-risk are cached_verified; fitting-source label audit is fresh_run. The data roles and 2% risk standard are unchanged.

Audited 108 groups. Easy labels per pair: 1,084 to 54,326. Easy prevalence among known rows: 16.412% to 37.469%. Minimum source easy count: 513.

Repeated-role row accesses: 5,741,442; known 5,614,596; unknown excluded from fitting 126,846. These are NOT independent examples.

## Fresh Development Readout

| Policy | ADE gain/floor (%) | Hard gain/floor (%) | Intervention fraction | Violating / undefined views |
|---|---:|---:|---:|---:|
| floor | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] | 0 / 216 |
| common_anchor | 0.00308501 [0.000905893, 0.00609838] | -0.000222994 [-0.000673015, 8.9018e-06] | 0.0108446 [0.00660599, 0.0153689] | 34 / 24 |
| raw_independent | 0.268809 [0.149047, 0.405042] | 0.254389 [0.112556, 0.418749] | 0.077985 [0.0584392, 0.0972454] | 65 / 16 |
| raw_joint | 0.507865 [0.263107, 0.790199] | 0.557981 [0.273242, 0.875534] | 0.077985 [0.0584392, 0.0972454] | 84 / 16 |
| raw_matched | 0.0236202 [0.0123851, 0.0361929] | 0.0216799 [0.00712515, 0.0392905] | 0.0108446 [0.00660599, 0.0153689] | 62 / 24 |
| marginal_independent | 0.432788 [0.256115, 0.632684] | 0.429637 [0.189261, 0.754968] | 0.126127 [0.098391, 0.155513] | 128 / 0 |
| marginal_joint | 0.847929 [0.46487, 1.28471] | 0.952733 [0.4749, 1.49751] | 0.126127 [0.098391, 0.155513] | 152 / 0 |
| marginal_matched | 0.0256072 [0.0140283, 0.0380816] | 0.0212159 [0.00620171, 0.038156] | 0.0108446 [0.00660599, 0.0153689] | 88 / 24 |
| supervised_independent | 0.000375205 [-0.0100254, 0.00724604] | 0.0212937 [0.000748219, 0.0554662] | 0.0131808 [0.00836199, 0.0180994] | 56 / 6 |
| supervised_joint | 0.00253887 [-0.00865641, 0.0102031] | 0.0243302 [0.00368049, 0.0569609] | 0.0131808 [0.00836199, 0.0180994] | 73 / 6 |
| supervised_matched | 0.00517293 [0.00212549, 0.0087337] | 0.00265059 [0.000320175, 0.00536957] | 0.0108446 [0.00660599, 0.0153689] | 51 / 24 |

## Prespecified Paired Contrasts

- marginal_joint_vs_raw_joint: ADE gain% 0.336364 [0.112251, 0.574232]; fixed-floor harm reduction pp -0.183142 [-0.303034, -0.0903288].
- marginal_matched_vs_raw_matched: ADE gain% 0.00197482 [-0.000231185, 0.00506549]; fixed-floor harm reduction pp -0.000966597 [-0.00206887, 0.000258981].
- supervised_joint_vs_raw_joint: ADE gain% -0.526243 [-0.825345, -0.268348]; fixed-floor harm reduction pp 0.142722 [0.0410423, 0.251742].
- supervised_matched_vs_marginal_matched: ADE gain% -0.0204671 [-0.0320605, -0.00956935]; fixed-floor harm reduction pp 0.00648741 [0.00216255, 0.0131405].
- supervised_matched_vs_raw_matched: ADE gain% -0.0184943 [-0.0301189, -0.00815552]; fixed-floor harm reduction pp 0.00552081 [0.0009513, 0.0128932].

## Easy-Risk Prediction

- marginal: Brier 0.262075 [0.218448, 0.317403]; log loss 1.09921 [0.824549, 1.53141]; signed MSE 0.00418997 [0.00245329, 0.00628605].
- supervised: Brier 0.158131 [0.144101, 0.173253]; log loss 0.499397 [0.454324, 0.546331]; signed MSE 0.00444302 [0.00267599, 0.00644676].

Exploratory screen pass: False. No deployment change.

## Boundaries

Independent selection/calibration/confirmation remain closed. The original incomplete primary is not replaced. Proper probability loss does not prove calibration, risk coverage or cross-domain generalization. No threshold search. Coordinates remain image-local detector silver; obs8/pred12 uses raw-frame stride12. No metric/seconds, human-gold, physical safety, true3D, foundation or submission-readiness claim. Stage5C/SMC disabled.
