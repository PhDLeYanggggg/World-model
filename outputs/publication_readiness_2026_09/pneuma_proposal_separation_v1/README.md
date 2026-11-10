# Athens: how far apart are the frozen forecasts?

Two policies can make the same number of edits at the same peak radius
without changing the whole scene by the same amount. I measured how far
apart the four frozen forecasts are before comparing them with any labels.
All 18 proposal packets and six forecast/past-roster pairs matched their
committed hashes. Each arm edits two distinct recipients and the peak
radius matches in every query.

| Date | Drone members | Windows before seed replication | Joint vs independent: different recipient sets / seed views | Full-grid ADE envelope | Relative-motion envelope |
| --- | ---: | ---: | ---: | ---: | ---: |
| 24 Oct | 1 | 107 | 289 / 321 | 0.006994 | 0.013842 |
| 29 Oct | 3 | 297 | 815 / 891 | 0.003932 | 0.007801 |
| 30 Oct | 1 | 120 | 318 / 360 | 0.006949 | 0.013745 |
| 1 Nov | 1 | 119 | 317 / 357 | 0.011066 | 0.021785 |

The last two columns bound the absolute difference the forecasts could
make to full-grid error for any common outcome. They are in the pilot
tangent-plane units, not estimates of physical accuracy. The bound is
the average forecast-to-forecast distance by actor and step, or by pair
and step for relative motion. By the reverse triangle inequality, this
is at least the absolute difference of the corresponding errors against
any shared labels. Aggregate by query, seed and member as in the planned
date grouping. This is not a bound on an available-only or edited-only
subgroup, a percentage gain, or a confidence interval.

Joint and independent differ on at least one forecast path in 1,759 of
1,929 views. Joint and additive differ in 1,730; joint and the
same-family restriction in only 130. The latter contrast has a date-mean
full-grid ADE envelope of 0.000156 to 0.000646. Joint and independent
also differ in mean edit-path displacement despite the common radius;
joint moves the edited paths by 0.00253 to 0.00393 more on average
across the four dates. The pointwise budget is matched, but a claim
about equal integrated edit magnitude would be incorrect.

This explains why a large change in recipient choice need not produce
a large whole-scene effect in a dense roster. The two later dates
alone would omit both the date with the smallest whole-scene envelope
and the only date with simultaneous drone members. The four-date
zero-shot design tests transfer across more acquisition dates; the
two-plus-two design tests calibration and release. The contrast between
them is a design trade-off, not a result. Neither date role was assigned
by this audit. The original 2% rule and prior decisions are unchanged.

The [readout](readout.json) also includes FDE envelopes, each date's
largest individual ADE envelope, and all comparisons. It contains no
outcomes. To replay from the verified past-roster and forecast packets:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_pneuma_proposal_separation.py
```
