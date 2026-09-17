# Training Objective and Scene Sampling Comparison

Fit-only, three historically used physical scenes. No independent confirmation or deployment.

| Arm | Gain vs CV (%) | Exploratory scene interval (%) | Positive held fits | Easy gate passes | Binary oracle (%) |
| --- | ---: | --- | ---: | ---: | ---: |
| row_log | -1.0269 | [-10.389052750214866, -0.11461665461081072] | 0/9 | 0/9 | 0.3242 |
| row_ade | -165.6747 | [-435.93173936860603, -2.088475791357425] | 0/9 | 0/9 | 1.2237 |
| scene_log | -1.4694 | [-10.819937482889696, -0.31191485617190207] | 0/9 | 0/9 | 0.3616 |
| scene_ade | -234.5298 | [-618.7674091754357, -4.585530619019074] | 0/9 | 0/9 | 1.0399 |
| scene_ade_harm | -237.2443 | [-627.1804046151824, -4.143832294665972] | 0/9 | 0/9 | 1.1382 |

Each fit has4,000updates. Oracle uses future labels for diagnosis only.
Primary, cohort and role boundaries unchanged; no metric/seconds or real-time causality claim.
