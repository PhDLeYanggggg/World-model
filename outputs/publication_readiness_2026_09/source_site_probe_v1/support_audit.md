# Source Physical-Site Support

`fresh_run` audit over `cached_verified` original SDD train40 assets. The source
admission and query population are unchanged. All current query agents are
pedestrians under the registered adapter. Local IDs are scoped by recording;
726 IDs do not certify726 distinct people across cameras.

| Held physical site | Complete windows | Positive / negative | Incomplete, unscored | Local IDs | Videos | Held positive rate | Complement training rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bookstore | 6944 | 3175 / 3769 | 1526 | 181 | 7 | 45.72% | 44.48% |
| coupa | 4250 | 2546 / 1704 | 395 | 79 | 4 | 59.91% | 41.34% |
| deathCircle | 3054 | 987 / 2067 | 1917 | 177 | 5 | 32.32% | 46.85% |
| gates | 1662 | 420 / 1242 | 932 | 78 | 7 | 25.27% | 46.44% |
| hyang | 6464 | 2911 / 3553 | 1690 | 211 | 13 | 45.03% | 44.80% |

Totals:22,374complete windows,10,039positive,12,335negative;6,460incomplete
stationary queries are not treated as negative. Every held physical site and
its training complement contain both classes. Each training fold excludes all
videos and scoped agent IDs of its held site. This is source-fit internal
cross-validation, not a new official held-out test or untouched confirmation.

The positive-rate shift is already substantial inside SDD. Therefore a constant
training-prior control and probability-mean decomposition are necessary: adding
RGB must not be confused with merely shifting the average probability. Rates
here describe support; they are not used to recalibrate held-site predictions.

Labels are any future annotation-center change, not verified intention or body
motion. Past annotation histories are supplied offline and may be interpolated;
this is not strict sensor-as-of prediction. Source observes8 and predicts12 at
stride12, +144rawframes. No metric or effective-seconds equivalence is asserted.
Main native8-to12 task and sealed development/calibration/confirmation remain
unchanged and closed. No Stage5C or SMC.
