# Evidence Gate

This is a controlled forecast experiment, not a deployment gate-count exercise.
The user's prospectively approved full-policy safety contract is unchanged.

| Check | Status | Evidence / boundary |
|---|---|---|
| Actual current-producer lineage identified | Pass | Later dimensionless producer separated from earlier family |
| Fresh neural training complete | Pass | 9 current + 18 earlier fits; 108,000 primary updates; observed exit 0 |
| Neutral and interruption replay | Pass | Both families exact; full 4,000-step neutral replays |
| Endpoints frozen before first readout | Pass | d2f8eda7 and 41f0d3f0 |
| Original metrics reproduce | Pass | 864 current-family values; earlier primary metrics exact |
| Paired readout replay | Pass | Both analysis JSON and record CSV hashes reproduced |
| Source-excluded training and causal inputs | Verified within scope | Audited/hash-verified existing data; no excluded rows drawn; no new raw-data audit claim |
| Current-producer overall accuracy lift | Fail | -0.929724 pp; conditional CI [-1.669310, -0.168497] |
| Current-producer hard accuracy lift | Fail | -1.679741 pp |
| Candidate easy ADE degradation <=2% | Fail | 8.433059% against CV |
| Candidate support on all fixed TRAIN views | Fail | 147/148; one unchanged required view remains unsupported |
| Complete intervention-policy 2% safety | Not established / not_run | No new candidate-cost head or independent calibration; partial labels are not zero |
| Independent confirmation | not_run | Independent roles remain unopened |
| Targeted regression tests | Pass | 54 passed; full historical suite not rerun |
| Deployment promotion | No | Existing deployment unchanged |
| Submission-ready evidence | No | Development-only negative accuracy result and unresolved safety |
| Stage5C execution / SMC | Off / Off | No execution or enablement |

Matched-count comparisons are utility-only. Empty matched risk remains
not_estimable; old failures are not rescored into passes. A label-aware support
oracle or output-bound relaxation is not a deployable model.
