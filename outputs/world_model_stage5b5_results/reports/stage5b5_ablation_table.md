| ablation | status | mean_target_improvement | note |
| --- | --- | --- | --- |
| baseline only | executed | 0.0 |  |
| linear residual | executed_numpy_fallback | -0.000499 |  |
| MLP residual | not_run_in_quick_mode | n/a |  |
| GRU history only | not_run_in_quick_mode | n/a | Torch GRU path hit local OMP/SHM failure |
| Transformer history only | not_run_in_quick_mode | n/a | Torch GRU path hit local OMP/SHM failure |
| GRU + neighbor interaction | not_run_in_quick_mode | n/a | Torch GRU path hit local OMP/SHM failure |
| Transformer + neighbor interaction | not_run_in_quick_mode | n/a | Torch GRU path hit local OMP/SHM failure |
| GRU + interaction + domain embedding | not_run_in_quick_mode | n/a | Torch GRU path hit local OMP/SHM failure |
| GRU + interaction + domain + hard-weighted training | not_run_in_quick_mode | n/a | Torch GRU path hit local OMP/SHM failure |
| horizon-conditioned decoder | executed_numpy_fallback | -0.000499 |  |
| hybrid direct + recurrent | not_run_in_quick_mode | n/a |  |
| residual gate enabled | executed_numpy_fallback | -0.000499 |  |
| map/scene context enabled | not_run_in_quick_mode | n/a |  |
