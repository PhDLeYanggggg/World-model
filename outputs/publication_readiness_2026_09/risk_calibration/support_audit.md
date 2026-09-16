# Calibration Support and Bound Sensitivity

Fresh metadata/array identity checks and analytical calculation; no future labels, training or calibration result.

Current reader: 9 recordings / 6 physical-scene groups; 0 unexposed-reviewed scenes; 0 assigned calibration scenes. Protocol is unapproved.

The table hypothetically treats all current scene groups as independent calibration units with zero observed loss. This is deliberately optimistic, not a legal data split or a claim of independence. Tolerances and delta are illustrative, not chosen for the experiment.

| Frozen policies | Risks | Tolerance | Upper bound with 6 zero-loss scenes | Zero-loss scene count needed by this bound |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.02 | 0.4996 | 3745 |
| 1 | 1 | 0.10 | 0.4996 | 150 |
| 1 | 1 | 0.20 | 0.4996 | 38 |
| 5 | 2 | 0.02 | 0.6645 | 6623 |
| 5 | 2 | 0.10 | 0.6645 | 265 |
| 5 | 2 | 0.20 | 0.6645 | 67 |
| 10 | 2 | 0.02 | 0.7066 | 7490 |
| 10 | 2 | 0.10 | 0.7066 | 300 |
| 10 | 2 | 0.20 | 0.7066 | 75 |
| 20 | 2 | 0.02 | 0.7464 | 8356 |
| 20 | 2 | 0.10 | 0.7464 | 335 |
| 20 | 2 | 0.20 | 0.7464 | 84 |

For a [0,1] scene loss, this implementation uses mean + sqrt(log(M*K/delta)/(2*n)), capped at 1. The sample requirement is the rearranged inequality at empirical loss zero. It is a limitation of this conservative bound, not an information-theoretic lower bound or evidence that every alternative calibration method requires that many scenes.

Window replication, bootstrap draws and more agents cannot be substituted for independent scenes. A bounded clipped-harm tolerance of 0.02 is not a 2% relative ADE/FDE guarantee; clipping and denominators differ. Changing risk definitions, units, population or query schedules requires protocol approval.

## Synthetic Dependence Diagnostic

20,000 independent simulations draw 6 Bernoulli scene losses with true risk 0.20, then copy each loss into 1,000 identical windows. At illustrative tolerance 0.10 and delta 0.05, the scene-level screen falsely accepts 0.00%; the invalid window-as-independent screen falsely accepts 26.67%. The analytical all-zero-scene probability is 0.8^6 = 26.2144%. This constructed example exposes pseudo-replication; it does not estimate real M3W failure rates or prove general calibration power.

No actual statistical risk certificate, untouched confirmation, deployment promotion or metric/seconds claim is produced.
