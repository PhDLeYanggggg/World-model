# Fitting Support and Fixed-Batch Diagnostics

Paired examples are repeated training draws, not independent observations.
Fixed batches are cloned from the original sampler and never used for selection.

| Candidate / event | Old / new valid pairs, first 40 batches per head | Old / new full-fit pairs | Fixed total loss decreased | Fixed ranking loss decreased |
|---|---:|---:|---:|---:|
| neural / all | 47019 / 47019 | 2355302 / 2355302 | 9/9 | 9/9 |
| neural / easy | 3471 / 11056 | 167291 / 541866 | 9/9 | 8/9 |
| damping097 / all | 36714 / 36715 | 1843620 / 1843624 | 9/9 | 8/9 |
| damping097 / easy | 2175 / 6266 | 104799 / 309201 | 7/9 | 7/9 |

More supervised pairs or lower fitting loss is not proof of better excluded-locality ordering.
Only the paired full and matched-count readout can establish that effect.
