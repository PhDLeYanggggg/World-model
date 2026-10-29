# Reference-Harm Forecast Readout

Source development only, not independent confirmation. All 18 endpoints fixed before readout.
8 observed / 12 predicted steps; stride 12 raw frames; image pixels, detector-track labels.

| Seed | Arm | ADE vs CV (%) | ADE vs train-selected (%) | Easy net degradation vs CV (%) | Easy positive harm vs CV (%) |
|---|---|---:|---:|---:|---:|
| 17 | original_ADE | 2.449067 | 4.338503 | 0.348641 | 15.694626 |
| 17 | reference_harm | 1.116468 | 3.317653 | -2.055545 | 13.234497 |
| 29 | original_ADE | 1.870192 | 3.830943 | 0.755262 | 16.137448 |
| 29 | reference_harm | 1.663816 | 3.762078 | -2.301002 | 12.993836 |
| 43 | original_ADE | 2.153530 | 4.171053 | 0.063307 | 15.477825 |
| 43 | reference_harm | 1.386864 | 3.525576 | -2.327986 | 12.982408 |
| mean_seed | original_ADE | 2.157596 | 4.113500 | 0.389070 | 15.720001 |
| mean_seed | reference_harm | 1.389049 | 3.535102 | -2.228178 | 13.054473 |

Mean seed paired locality difference: -0.768547 pp; 3,000 locality-bootstrap interval [-2.281845531219107, 0.832022813787061].
This interval is conditional exploratory uncertainty, not independent confirmation.
Positive harm and net degradation are separate. No full-policy safety or deployment claim.
Missing future labels and zero reference percentages remain explicit; every recording is retained.
No Stage5C, SMC, metric, seconds, true-3D or foundation-model claim.
