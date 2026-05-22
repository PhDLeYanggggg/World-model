# Stage 13 Failure Analysis

- failure_case_count: `80`
- worse_than_baseline_count: `26`
- easy_degraded_count: `23`
- alpha_high_easy_count: `36`
- t100_drift_count: `0`
- interaction_failure_count: `130`

| failure_type | dataset | subset | horizon | improvement | likely reason |
| --- | --- | --- | --- | --- | --- |
| baseline_better_than_model | eth_ucy | all | 1 | -0.104415 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | baseline_failure | 1 | -0.102717 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | ge5 | 1 | -0.119964 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | goalbench_official | 1 | -0.104415 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | hard | 1 | -0.105066 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | aerialmpt | all | 1 | -0.132638 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | aerialmpt | baseline_failure | 1 | -0.132638 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | aerialmpt | ge5 | 1 | -0.132638 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | aerialmpt | goalbench_official | 1 | -0.132638 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | aerialmpt | hard | 1 | -0.132638 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | all | 1 | -0.218594 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | baseline_failure | 1 | -0.215602 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | easy | 1 | -0.174608 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | ge5 | 1 | -0.251134 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | goalbench_official | 1 | -0.218594 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy | hard | 1 | -0.220347 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy_ewap | all | 1 | -0.112582 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy_ewap | baseline_failure | 1 | -0.112582 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy_ewap | ge5 | 1 | -0.112582 | Model correction made FDE worse than baseline by >10%. |
| baseline_better_than_model | eth_ucy_ewap | goalbench_official | 1 | -0.112582 | Model correction made FDE worse than baseline by >10%. |
| easy_degraded | eth_ucy | easy | 1 | -0.088078 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.174608 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 5 | -0.024905 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.050832 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.04737 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.056072 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.052247 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.041902 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.040101 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.067293 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.062755 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.043026 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.040101 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.047675 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.04737 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.05638 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.052559 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.050832 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.04737 | Easy subset degraded beyond preservation tolerance. |
| easy_degraded | eth_ucy | easy | 1 | -0.065711 | Easy subset degraded beyond preservation tolerance. |
| alpha_high_on_easy | eth_ucy | easy | 1 | -0.088078 | Alpha intervention is too high on easy samples. |
| alpha_high_on_easy | eth_ucy | easy | 5 | -0.016728 | Alpha intervention is too high on easy samples. |
| alpha_high_on_easy | eth_ucy | easy | 10 | -0.005606 | Alpha intervention is too high on easy samples. |
| alpha_high_on_easy | eth_ucy | easy | 1 | -0.174608 | Alpha intervention is too high on easy samples. |
| alpha_high_on_easy | eth_ucy | easy | 5 | -0.024905 | Alpha intervention is too high on easy samples. |
| alpha_high_on_easy | eth_ucy | easy | 10 | -0.0115 | Alpha intervention is too high on easy samples. |
| alpha_high_on_easy | eth_ucy | easy | 1 | -0.050832 | Alpha intervention is too high on easy samples. |
| alpha_high_on_easy | eth_ucy | easy | 5 | -0.007536 | Alpha intervention is too high on easy samples. |
| alpha_high_on_easy | eth_ucy | easy | 10 | -0.004738 | Alpha intervention is too high on easy samples. |
| alpha_high_on_easy | eth_ucy | easy | 1 | -0.056072 | Alpha intervention is too high on easy samples. |
