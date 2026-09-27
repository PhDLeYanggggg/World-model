# Fixed-Budget Training

| Trial | Parameters | Updates | Draws | Distinct fitting queries | Held draws | Fit seconds | Last logged loss |
|---|---:|---:|---:|---:|---:|---:|---:|
| single0_seed17 | 88514 | 4000 | 256000 | 60181 | 0 | 58.300 | 0.852888 |
| single0_seed29 | 88514 | 4000 | 256000 | 60531 | 0 | 57.918 | 0.966489 |
| single0_seed43 | 88514 | 4000 | 256000 | 60331 | 0 | 62.412 | 1.210373 |
| single1_seed17 | 88514 | 4000 | 256000 | 23754 | 0 | 63.426 | 1.221602 |
| single1_seed29 | 88514 | 4000 | 256000 | 23755 | 0 | 66.219 | 0.994150 |
| single1_seed43 | 88514 | 4000 | 256000 | 23758 | 0 | 68.383 | 0.695074 |
| single2_seed17 | 88514 | 4000 | 256000 | 69367 | 0 | 60.439 | 0.620645 |
| single2_seed29 | 88514 | 4000 | 256000 | 69873 | 0 | 59.211 | 1.043111 |
| single2_seed43 | 88514 | 4000 | 256000 | 69562 | 0 | 55.434 | 0.687821 |

The first 100 updates resume into the first 4,000-update endpoint, not a second budget.
All paired initial parameters and sampling states/counts match the grouped controls.
The last training-batch loss is not a validation score. All nine endpoints are retained.
Cumulative fresh training seconds: 551.743.
No new control fitting; cached controls have hash and full fresh inference verification.
