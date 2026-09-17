# Frozen Appearance Support Controls

No new training or model selection. All 18 models and four treatments retained.

| Held scene | Input | Treatment | Unrestricted gain % | Guarded gain % | Easy absolute harm | Switch rate |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ETH | geometry | original | -0.39145 | -0.01330 | 0.000000 | 1.2346% |
| ETH | geometry | jacobian_box | -0.43082 | -0.01134 | 0.000000 | 1.2346% |
| ETH | geometry | all_feature_box | -1.15999 | -0.03296 | 0.176733 | 2.4691% |
| ETH | geometry | support_fallback | -0.39145 | 0.00000 | 0.000000 | 0.0000% |
| ETH | current_rgb | original | -7.69380 | -8.09403 | 41.502993 | 65.8436% |
| ETH | current_rgb | jacobian_box | -8.07638 | -8.34924 | 42.467172 | 65.8436% |
| ETH | current_rgb | all_feature_box | -9.34410 | -8.57067 | 43.243502 | 65.8436% |
| ETH | current_rgb | support_fallback | -7.69380 | 0.00000 | 0.000000 | 0.0000% |
| ETH | past_rgb | original | -14.52907 | -14.97607 | 79.014463 | 66.6667% |
| ETH | past_rgb | jacobian_box | -15.15970 | -15.50018 | 81.059558 | 66.6667% |
| ETH | past_rgb | all_feature_box | -16.85845 | -16.14611 | 83.600248 | 66.6667% |
| ETH | past_rgb | support_fallback | -14.52907 | 0.00000 | 0.000000 | 0.0000% |
| Hotel | geometry | original | -513.47859 | -132.03553 | 30.219201 | 20.5399% |
| Hotel | geometry | jacobian_box | -556.24394 | -98.12424 | 20.200145 | 14.3192% |
| Hotel | geometry | all_feature_box | -444.78743 | -60.34225 | 15.454497 | 13.0282% |
| Hotel | geometry | support_fallback | -513.47859 | 0.00000 | 0.000000 | 0.0000% |
| Hotel | current_rgb | original | -389.04297 | -167.72675 | 36.980879 | 35.2113% |
| Hotel | current_rgb | jacobian_box | -331.48886 | -110.24010 | 25.753957 | 28.4038% |
| Hotel | current_rgb | all_feature_box | -266.71318 | -90.80889 | 20.851913 | 26.0563% |
| Hotel | current_rgb | support_fallback | -389.04297 | 0.00000 | 0.000000 | 0.0000% |
| Hotel | past_rgb | original | -401.79574 | -172.63225 | 38.415231 | 35.6808% |
| Hotel | past_rgb | jacobian_box | -350.42502 | -110.85775 | 24.983505 | 29.3427% |
| Hotel | past_rgb | all_feature_box | -266.23230 | -82.86013 | 19.187921 | 26.1737% |
| Hotel | past_rgb | support_fallback | -401.79574 | 0.00000 | 0.000000 | 0.0000% |

Easy percentage ratios are undefined at the zero CV floor. No marginal box certifies joint support.
Native ADE/FDE and all seeds remain in metrics.json. No independent test/scene CI or deployment.
