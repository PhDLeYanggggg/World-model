# WEPDTOF: history quality and the velocity reference

I wanted to know why four-point velocity was harder to beat on
[WEPDTOF](https://openaccess.thecvf.com/content/WACV2022/html/Tezcan_WEPDTOF_A_Dataset_and_Benchmark_Algorithms_for_In-the-Wild_People_Detection_WACV_2022_paper.html).
The original readout used a detector-driven roster. I reconstructed
the annotation history only when the same track-to-person match
holds at all eight observed steps. This retains 353 of 431 online
track-windows across 15 clips in 13 sources, with 4,209 observed
future positions and 348 endpoints. All cells use the same positions
and masks. Annotation inputs are only a diagnostic.

| Past input | CV minus OLS4 ADE (px) | 95% source bootstrap |
| --- | ---: | ---: |
| Detector anchor and velocity | 5.98 | [-2.37, 15.47] |
| Annotation anchor only | 5.93 | [-2.40, 15.44] |
| Annotation velocity only | 3.28 | [-4.56, 13.50] |
| Annotation anchor and velocity | 2.99 | [-5.06, 13.45] |

The CV-to-OLS4 gap shrinks by **2.99 px** [−0.58, 6.27] when both inputs
come from the annotations. The point estimate suggests that
observation noise accounts for part of OLS4's advantage, especially
in velocity, but the gap reduction is not resolved by this cohort.
Replacing the whole history reduces CV ADE by **7.44 px** [4.71, 10.25]
and OLS4 by **4.45 px** [1.84, 6.55]. Averaging the two replacement
orders attributes 2.85 and 4.59 px of the CV reduction to the last
position and velocity respectively; for OLS4, 2.68 and 1.77 px.
This isolates effects in the forecast formula; it does not identify
the causes of detector error or represent missing tracks.

I also tested a causal, detector-only change: keep the same OLS4
slope, but extrapolate from its fitted last position instead of the
last detection. Its paired ADE gain over OLS4 is **−0.012 px**
[−0.573, +0.630], and FDE gain is +0.038 px [−0.509, +0.642].
Only 1 of 15 clip views passes the original CV-relative risk check;
four have undefined easy reference and worst estimable signed easy
degradation is 137.2%. I kept the frozen references and policy.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_wepdtof_history_factorial.py
```

`fresh_run`: retrospective scoring on the already opened WEPDTOF
annotations. `cached_verified`: all frozen CV/OLS4 forecasts, labels,
original match counts and original ADE/FDE/p95 receipts at every seed.
The new baselines are deterministic, so the three original seeds
are not counted as new evidence. Intervals use 3,000 paired source
resamples after averaging within source. `not_run`: independent
evaluation on a new cohort, training, CREATE, Stage5C and SMC.
No thresholds, deployed forecasts or original outcomes changed.

[Protocol](protocol.md) | [Readout](readout.json).
