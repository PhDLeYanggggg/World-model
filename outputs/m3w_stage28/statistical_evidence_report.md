# Stage28 Statistical Evidence Report

- Uses 1000 bootstrap resamples on the held-out test rows.
- Reports pixel-space raw-frame metrics; no metric or seconds-level claim.

| subset | mean improvement | 95% CI | n |
| --- | ---: | --- | ---: |
| official_t50 | 0.168513 | [0.159876, 0.177258] | 24810 |
| hard_failure | 0.133687 | [0.128161, 0.138759] | 96581 |
| all | 0.132510 | [0.127405, 0.138045] | 100000 |
| within_scene | 0.158141 | [0.150144, 0.165435] | 50000 |
| cross_scene | 0.106126 | [0.099369, 0.113080] | 50000 |

- easy degradation point estimate: `0.01928694490688554`
