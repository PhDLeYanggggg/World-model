# Source-Only Cost-Head Fitting Logs

Source: the frozen analysis.json. No external inference or evaluation is performed by this exporter.

The neural entries are losses on different sampled minibatches, not a fixed validation set.
Forest entries are draw-count-weighted fitting MSE over sampled source rows. These columns
are not directly comparable performance estimates; no head or seed is selected from them.

| Forecast family / seed | Neural step 1 | Neural step 3000 | Forest 16 trees | Forest 128 trees | Fit seconds, both heads |
|---|---:|---:|---:|---:|---:|
| eqmotion_seed17 | 0.141841 | 0.105823 | 0.096980 | 0.096402 | 36.378 |
| eqmotion_seed29 | 0.143961 | 0.095702 | 0.096332 | 0.096014 | 37.452 |
| eqmotion_seed43 | 0.137413 | 0.110345 | 0.097187 | 0.097263 | 37.230 |
| transformer_seed17 | 0.118648 | 0.090814 | 0.091931 | 0.091184 | 32.887 |
| transformer_seed29 | 0.120095 | 0.089844 | 0.090338 | 0.089882 | 32.909 |
| transformer_seed43 | 0.123928 | 0.101923 | 0.090791 | 0.090300 | 32.614 |

All six final minibatch losses are below their step-1 values, but that is not
convergence or downstream-lift evidence. The EqMotion seed-43 forest's final fitting MSE
is slightly higher than at sixteen trees. All endpoints are kept regardless of this sign.

Summed fit time: 209.470084 seconds. This excludes source loading,
OOF target/feature construction, verification and the already-completed forecasting fits.
The 234 complete log records are in training_loss.csv. No metric/seconds or independent safety claim.
