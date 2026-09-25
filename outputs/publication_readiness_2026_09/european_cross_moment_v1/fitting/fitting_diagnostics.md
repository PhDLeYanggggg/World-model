# Fitting Support and Fixed-Batch Diagnostics

Paired examples are repeated training draws, not independent observations.
Weight effective pair count is (sum w)^2/sum(w^2), not an independent sample size.
Fixed batches are cloned from the original sampler and never used for selection.

| Candidate / event | Share / cross pairs, first 40 fitting batches | Control / treatment full-fit pairs | Fixed total loss decreased | Fixed ranking loss decreased |
|---|---:|---:|---:|---:|
| neural / all | 47019 / 47019 | 2355302 / 2355302 | 9/9 | 7/9 |
| neural / easy | 11056 / 11056 | 541866 / 541866 | 7/9 | 6/9 |
| damping097 / all | 36715 / 36715 | 1843624 / 1843624 | 9/9 | 8/9 |
| damping097 / easy | 6266 / 6266 | 309201 / 309201 | 8/9 | 7/9 |

Audit old/new counts compare share/cross targets in both modes, not the preceding normalizer.
A lower fitting loss is not proof of better excluded-locality ordering.
Only the paired full and matched-count readout can establish that effect.
