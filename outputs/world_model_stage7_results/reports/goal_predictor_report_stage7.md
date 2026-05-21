# Stage 7 Goal / Intent Predictor

| split | samples | top1_goal_accuracy | top3_goal_accuracy | goal_NLL | goal_ECE | goal_entropy | majority_top1 | majority_top3 | beats_majority_top3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 75 | 0.64 | 0.986667 | 0.976554 | 0.1937 | 0.987953 | 0.48 | 0.906667 | True |
| val | 20 | 0.45 | 0.85 | 1.161674 | 0.07814 | 1.127156 | 0.35 | 0.9 | False |
| test | 23 | 0.434783 | 0.782609 | 1.375813 | 0.173383 | 1.103017 | 0.304348 | 0.826087 | False |
| test:eth_ucy | 6 | 0.833333 | 1.0 | 0.427941 | 0.255095 | 0.826137 | 0.833333 | 1.0 | False |
| test:tgsim | 4 | 0.0 | 0.5 | 2.467362 | 0.447312 | 1.251994 | 0.75 | 1.0 | False |
| test:tgsim_i90 | 6 | 0.5 | 1.0 | 1.072968 | 0.405424 | 1.090163 | 0.5 | 1.0 | False |
| test:trajnet | 7 | 0.285714 | 0.571429 | 1.824115 | 0.097714 | 1.266232 | 0.714286 | 1.0 | False |

