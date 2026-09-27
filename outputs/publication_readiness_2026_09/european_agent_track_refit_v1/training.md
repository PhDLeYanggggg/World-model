# Fixed-Budget Training

| Trial | Parameters | Updates | Draws | Distinct fitting queries | Held draws | Fit seconds | Last logged loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| single0_seed17 | 88514 | 4000 | 256000 | 60181 | 0 | 56.225 | 0.864393 |
| single0_seed29 | 88514 | 4000 | 256000 | 60531 | 0 | 62.735 | 1.049055 |
| single0_seed43 | 88514 | 4000 | 256000 | 60331 | 0 | 63.684 | 1.226176 |
| single1_seed17 | 88514 | 4000 | 256000 | 23754 | 0 | 67.091 | 1.141885 |
| single1_seed29 | 88514 | 4000 | 256000 | 23755 | 0 | 67.868 | 1.095938 |
| single1_seed43 | 88514 | 4000 | 256000 | 23758 | 0 | 68.924 | 0.765147 |
| single2_seed17 | 88514 | 4000 | 256000 | 69367 | 0 | 67.654 | 0.618262 |
| single2_seed29 | 88514 | 4000 | 256000 | 69873 | 0 | 67.631 | 1.136628 |
| single2_seed43 | 88514 | 4000 | 256000 | 69562 | 0 | 68.467 | 0.770058 |

The first 100 updates resume into the first 4,000-update endpoint, not a second budget.
All paired initial parameters and sampling states/counts match the flat controls.
The last training-batch loss is not a validation score. All nine endpoints are retained.
Cumulative fresh training seconds: 590.279.
No new control fitting; cached controls have hash and full fresh inference verification.
