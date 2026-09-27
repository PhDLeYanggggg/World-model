# Training Completion

## Material Passport

Fresh216 risk-head fits. Fixed training monitors are not validation loss or downstream evidence.

| Loss | Heads | Updates | Cumulative fit seconds | Unknown draws | Heads with lower fixed MSE | Heads with lower fixed tail loss | Mean initial/final MSE |
|---|---:|---:|---:|---:|---:|---:|---:|
| mse | 108 | 216000 | 151.97 | 0 | 108 | 108 | 0.713982 / 0.340475 |
| tail4 | 108 | 216000 | 158.40 | 0 | 108 | 108 | 0.713992 / 0.349089 |

Paired heads use identical initial models, fitting features, targets, normalizers, draws and RNG states.
Only loss weights differ. The first100-update pilot is included in the2,000-update budget.
Forecasters, floor producer chain and ridge utility remain frozen. No independent-source tuning.
