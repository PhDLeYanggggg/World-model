# Protected Causal Motion Controls

Fresh source-development readout; no external selection or deployment. Twelve cached-verified
Transformer heads; 72 fresh causal heads; 84 fresh matched forests. Native annotation pixels,
8/12 native steps. Four previously exposed sites. CIs bootstrap sites, not overlapping windows.

| Action | Head | Strict ADE gain % [CI95] | Hard gain % | Worst site/seed easy degradation % | Switch % |
|---|---|---:|---:|---:|---:|
| constant_position | neural | 1.8167 [0.9346, 3.3695] | 0.6202 | 2.0551 | 3.345 |
| constant_position | forest | 0.7484 [0.2934, 1.4599] | 0.0004 | 0.3484 | 1.627 |
| damped_velocity_005 | neural | 3.6347 [2.6423, 4.6860] | 4.7232 | 2.4944 | 12.252 |
| damped_velocity_005 | forest | 1.4144 [0.8491, 2.1612] | 1.3844 | 0.0000 | 3.689 |
| damped_velocity_010 | neural | 3.7447 [2.4209, 5.4592] | 4.4381 | 3.0383 | 8.550 |
| damped_velocity_010 | forest | 1.2763 [0.7236, 1.8290] | 0.8178 | 0.0000 | 3.197 |
| damped_velocity_020 | neural | 3.7059 [2.2530, 5.5713] | 3.6515 | 3.7374 | 6.020 |
| damped_velocity_020 | forest | 1.1048 [0.3930, 2.1732] | 0.2343 | 0.0000 | 2.894 |
| constant_acceleration_causal | neural | -0.0014 [-0.0028, -0.0001] | -0.0007 | 0.0005 | 0.001 |
| constant_acceleration_causal | forest | 0.0000 [0.0000, 0.0000] | 0.0000 | 0.0000 | 0.000 |
| constant_turn_rate | neural | 0.5635 [0.2704, 0.8566] | 0.7478 | 1.4846 | 0.559 |
| constant_turn_rate | forest | 0.0039 [0.0008, 0.0080] | 0.0000 | 0.0041 | 0.012 |
| transformer | neural | 2.4368 [1.6792, 3.1114] | 2.2918 | 1.0669 | 6.676 |
| transformer | forest | 1.3020 [0.5764, 2.0276] | 0.5390 | 0.1757 | 3.251 |

## Paired Incremental Forecast Contribution

Positive means full Transformer improves more than the protected simple action.
No best arm is selected from this table. Small four-site intervals are exploratory.

| Simple action | Head | Transformer minus action pp [CI95] | Matched-count pp [CI95] |
|---|---|---:|---:|
| constant_position | neural | 0.6202 [-0.7135438097316895, 1.8513832998991648] | 0.1838 [-0.9933762047929531, 1.156519383557311] |
| constant_position | forest | 0.5536 [0.09294824893319453, 1.161077675439931] | 0.1101 [-0.3512659058327733, 0.6920087861762098] |
| damped_velocity_005 | neural | -1.1979 [-1.623440622202732, -0.7722714366538763] | -0.7584 [-1.3253941756133263, -0.2111995772308195] |
| damped_velocity_005 | forest | -0.1124 [-0.7386005855613748, 0.6739625442474528] | -0.0877 [-0.7566665652464527, 0.7363326701433526] |
| damped_velocity_010 | neural | -1.3079 [-2.3178390463477356, -0.6764314539106253] | -1.0594 [-2.25854117082552, -0.2786752980339785] |
| damped_velocity_010 | forest | 0.0257 [-0.1950669461713045, 0.3404189041074007] | 0.0015 [-0.24534309411811117, 0.3455431690323113] |
| damped_velocity_020 | neural | -1.2690 [-2.492273796217112, -0.49072035918291124] | -1.2337 [-2.622514174429838, -0.4585972684522932] |
| damped_velocity_020 | forest | 0.1972 [-0.3037339299726993, 1.0028914595118954] | 0.1224 [-0.31519953732385697, 0.7975244862902853] |
| constant_acceleration_causal | neural | 2.4383 [1.681645209589211, 3.1130902863795717] | 0.0084 [-0.0010837746056147068, 0.01793360385522469] |
| constant_acceleration_causal | forest | 1.3020 [0.5764057914706411, 2.027553222656847] | 0.0000 [0.0, 0.0] |
| constant_turn_rate | neural | 1.8733 [1.2001423674787126, 2.546488055411966] | 0.4397 [0.12850387252634765, 0.7711326388826295] |
| constant_turn_rate | forest | 1.2981 [0.5701844510316967, 2.0259497884197506] | 0.0276 [0.0016400754926759742, 0.053656733074586294] |

## Boundaries

- Equal mean scene-relative available-label ADE; mean seed errors are not an ensemble forecast.
- Complete, partial, zero-CV, easy/hard, FDE, tails, counts and missing-label bounds are in analysis.json.
- Easy mean preservation does not establish individual or population safety.
- Cost learner is neural even for a simple action; this separates forecasting from routing contribution.
- No images, goal contribution, joint-scene contribution, metric calibration or seconds claim.
- DUT is not re-evaluated; DroneCrowd remains closed; nested EqMotion controls are not_run here.
- Existing deployment is unchanged. Stage5C execution and SMC remain disabled.
