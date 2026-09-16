# Fit-Only Training-Scale Audit

`fresh_run`; no development/confirmation labels, training changes or sample deletion.

| Fit recording | Rows | At scale floor | Largest ~1% target-energy share | Max target square |
| --- | ---: | ---: | ---: | ---: |
| eth_eth | 2614 | 81 | 96.32% | 1088354.34 |
| eth_hotel | 1197 | 284 | 49.81% | 42524.33 |
| ucy_zara01 | 2234 | 0 | 57.23% | 167.51 |
| ucy_zara02 | 5741 | 0 | 99.74% | 445108.93 |
| ucy_zara03 | 180 | 0 | 97.47% | 10827.31 |

The past-derived scale is max(history path length, last speed times horizon, 0.001).
Stationary-to-moving cases can therefore have extremely large normalized targets.
This is a plausible explanation for volatile squared-loss fitting, not proof of the full failure mechanism.
It measures target energy, not actual per-example gradient or realized prediction error.
Do not discard these cases or renormalize test errors after seeing scores.
First compare a registered robust-loss fit on the same inputs, then audit a train-derived scale floor separately.
