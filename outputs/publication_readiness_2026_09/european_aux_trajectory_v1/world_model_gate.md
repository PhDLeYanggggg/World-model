# Evidence Status After Training-Trajectory Reconstruction

| Item | Status | Evidence boundary |
|---|---|---|
| Fixed-budget real Torch reconstruction | Complete | 432 heads, 864000 updates |
| Exact original numerical states | Pass | All432 final states and prescribed traces match |
| Intermediate observation coverage | Complete | 2160 snapshots; all5 times and3 arms retained |
| Fitting-role boundaries | Preserved | Hash-bound inputs/targets; no new outer outcome use |
| Auxiliary cost improvement | Not supported | Every full and motion cost4 point is worse than cost-only |
| Stable easy-harm benefit over cost-only | Not supported | No full-input wholly positive interval at any time |
| Early auxiliary task information | Diagnostic only | Some true-vs-shuffled benefits do not beat cost-only |
| Prior-initialization explanation | Hypothesis only | Post-hoc fitting metadata, not a causal intervention |
| Repaired initialization model | not_run | Must be separately registered and trained |
| New held-scene trajectory/policy improvement | not_run | Fitting curves cannot establish it |
| Independent selection/calibration/confirmation | Unopened | No data-role relabeling |
| Deployment promotion | No | Original scientific failure remains |
| Submission readiness | No | Method, independent evidence and paper gaps remain |
| Stage5C execution | False | Disabled |
| SMC | False | Disabled |

Engineering replay status is recorded separately in verification.json after
completion; it cannot turn scientific failure into success. The420 unsupported
stratum cells remain explicitly not_estimable. Intervals use3000 descriptive
four-locality resamples, not independent scene replication. Obs8/pred12 detector
pixels only; no metric/seconds, human-gold, physical-safety, true3D or foundation
claim. No checkpoint selection from training curves.
