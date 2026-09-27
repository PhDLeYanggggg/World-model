# Logged Gradient Diagnostic

Same logged steps, initial loss, sample draws and clipping threshold. All nine trials retained.
Logs sample every50updates plus the initial step, not all4,000 gradients.
This comparison cannot isolate optimization changes from coordinate-unit invariance.
Lower training loss or fewer clipped logged gradients is not held-out forecast improvement.

| Trial | Arm | Logged steps | Above clipping threshold | Median gradient norm | Final logged loss |
|---|---|---:|---:|---:|---:|
| single0_seed17 | dimensionless | 81 | 0 | 1.228120 | 0.852888 |
| single0_seed17 | grouped | 81 | 81 | 151.709564 | 0.864393 |
| single0_seed29 | dimensionless | 81 | 0 | 1.150909 | 0.966489 |
| single0_seed29 | grouped | 81 | 81 | 135.095261 | 1.049055 |
| single0_seed43 | dimensionless | 81 | 0 | 1.207773 | 1.210373 |
| single0_seed43 | grouped | 81 | 81 | 111.528931 | 1.226176 |
| single1_seed17 | dimensionless | 81 | 0 | 0.972365 | 1.221602 |
| single1_seed17 | grouped | 81 | 81 | 85.183388 | 1.141885 |
| single1_seed29 | dimensionless | 81 | 0 | 0.966371 | 0.994150 |
| single1_seed29 | grouped | 81 | 81 | 101.962486 | 1.095938 |
| single1_seed43 | dimensionless | 81 | 0 | 0.933958 | 0.695074 |
| single1_seed43 | grouped | 81 | 81 | 90.845818 | 0.765147 |
| single2_seed17 | dimensionless | 81 | 2 | 2.275918 | 0.620645 |
| single2_seed17 | grouped | 81 | 81 | 231.125900 | 0.618262 |
| single2_seed29 | dimensionless | 81 | 1 | 2.526031 | 1.043111 |
| single2_seed29 | grouped | 81 | 81 | 202.290512 | 1.136628 |
| single2_seed43 | dimensionless | 81 | 0 | 2.627799 | 0.687821 |
| single2_seed43 | grouped | 81 | 81 | 222.139816 | 0.770058 |
