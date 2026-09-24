# Training Loss and Compute

These are training-minibatch losses, not validation or held-site metrics. No loss-based checkpoint selection was used.
The objective is masked native ADE divided by a fitting-site baseline normalizer, with indexed/supported sampling correction.
The inherited trace field `mean_past_normalized_ADE` actually stores the unweighted native-pixel minibatch ADE for this input path; its legacy name must not be interpreted as a normalized score.

| Trial | Steps | Parameters | Fitting seconds | First logged loss | Mean last 5 logged losses | Held rows drawn |
|---|---:|---:|---:|---:|---:|---:|
| single0_seed17 | 4000 | 88514 | 74.92 | 1.112890 | 0.949503 | 0 |
| single0_seed29 | 4000 | 88514 | 77.75 | 1.275578 | 0.854762 | 0 |
| single0_seed43 | 4000 | 88514 | 78.26 | 1.065716 | 1.086321 | 0 |
| single1_seed17 | 4000 | 88514 | 77.68 | 0.862717 | 1.074737 | 0 |
| single1_seed29 | 4000 | 88514 | 132.83 | 0.700135 | 1.019173 | 0 |
| single1_seed43 | 4000 | 88514 | 175.73 | 1.235687 | 0.947452 | 0 |
| single2_seed17 | 4000 | 88514 | 123.37 | 0.833574 | 0.734449 | 0 |
| single2_seed29 | 4000 | 88514 | 117.65 | 1.095522 | 0.903733 | 0 |
| single2_seed43 | 4000 | 88514 | 126.01 | 0.696007 | 0.987574 | 0 |
| complement0_seed17 | 4000 | 88514 | 125.98 | 0.891544 | 0.899028 | 0 |
| complement0_seed29 | 4000 | 88514 | 123.49 | 1.145915 | 0.921570 | 0 |
| complement0_seed43 | 4000 | 88514 | 123.44 | 1.173694 | 0.971198 | 0 |
| complement1_seed17 | 4000 | 88514 | 123.59 | 1.325479 | 0.965875 | 0 |
| complement1_seed29 | 4000 | 88514 | 77.03 | 1.026473 | 0.746172 | 0 |
| complement1_seed43 | 4000 | 88514 | 75.15 | 0.945703 | 0.788734 | 0 |
| complement2_seed17 | 4000 | 88514 | 75.53 | 1.047419 | 0.871932 | 0 |
| complement2_seed29 | 4000 | 88514 | 74.14 | 0.869658 | 0.873929 | 0 |
| complement2_seed43 | 4000 | 88514 | 133.13 | 1.080946 | 0.978907 | 0 |

Total: 18 fits, 72000 updates, 1915.68 summed fitting seconds. This excludes cache creation, inference and verification.
Native arm64 CPU4/inter-op1/workers0. The real first 100 updates were resumed inside the fixed budget. No CUDA/MPS claim or NumPy substitute.
