# Source-Only Training Losses

The fixed endpoints were chosen before training, not by these losses.
Loss is a sampled minibatch native-coordinate objective with source-fit normalization.
The first/last summaries average ten logged batches, not epochs or held-out data.
They do not establish convergence, external accuracy, safety or comparative superiority.

| Predictor | Seed | Updates | Seconds | First logged loss | Final logged loss | First 10 mean | Last 10 mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| transformer | 17 | 4000 | 82.08 | 1.185601 | 0.928229 | 0.979608 | 0.993403 |
| transformer | 29 | 4000 | 83.70 | 0.964696 | 0.879987 | 1.076313 | 0.982377 |
| transformer | 43 | 4000 | 83.37 | 1.628541 | 1.037830 | 0.964438 | 0.918103 |
| eqmotion | 17 | 4000 | 1405.95 | 18.984056 | 0.906030 | 3.300715 | 0.990486 |
| eqmotion | 29 | 4000 | 1408.01 | 18.839554 | 0.858711 | 3.380025 | 0.975422 |
| eqmotion | 43 | 4000 | 1410.08 | 20.922634 | 0.992137 | 3.415031 | 0.890185 |

Full logged batches: [training_loss.csv](training_loss.csv).
The reported gradient norm is measured before clipping at 5, not the final update norm.
The separate past-normalized ADE debug field is not this objective: very small past
motion denominators can make it large. Neither field is an external performance metric.
All reserved-source predictions, independent calibration and confirmation remain not_run.

Analysis SHA256: `494af734ba3ddd9f18369c7485e28252d860b804e7db81776b1bc77713beb0a5`.
