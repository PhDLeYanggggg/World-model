# Fit-Only Motion-Bound Headroom

Only registered fit rows. The oracle uses future labels for diagnosis, never as inference inputs.
It chooses an arbitrary correction independently at each requested step. This is an optimistic infimum inside each correction ball, not a trained network or a generalization bound.

| Recording | Rows | CV normalized ADE | Oracle ADE infimum | Optimistic headroom % | Zero-budget rows | Zero-budget share of CV error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eth_eth | 2614 | 7.93099 | 7.79306 | 1.739065444368317 | 81 | 0.9766473781105302 |
| eth_hotel | 1197 | 5.27697 | 5.09443 | 3.459260104525309 | 284 | 0.9516712969036033 |
| ucy_zara01 | 2234 | 0.178831 | 0.0635998 | 64.43587851677873 | 0 | 0.0 |
| ucy_zara02 | 5741 | 1.4382 | 1.33291 | 7.321350310365274 | 0 | 0.0 |
| ucy_zara03 | 180 | 0.791297 | 0.653793 | 17.37699720071318 | 0 | 0.0 |

Exactly stationary pasts with a stationary causal baseline have zero correction radius. Any later movement remains scored; these rows are not removed.
The upper bound ignores learnability, shared network parameters and temporal smoothness. Large theoretical headroom does not prove a predictor can capture it. Small headroom does not authorize changing this running experiment.
All values retain the registered past normalization. No physical-distance, seconds, calibrated-safety or deployment claim. Neither the training configuration nor the primary metric changes.
