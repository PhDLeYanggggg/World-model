# Incumbent-Relative Intervention: Complete Development Results

I trained144 Torch heads and72 ridge controls on the same four-source forecasts and381 causal inputs.
Only the learning reference and corresponding fallback rule change. All288 registered decisions froze before new readout.
Three seeds, six ordered source rotations and two event targets are dependent development views, not independent tests.

| Policy | All ADE gain vs incumbent (%) | Positive / negative CI | Hard gain vs incumbent (%) | Worst easy degradation vs CV (%) | Override rate |
|---|---:|---:|---:|---:|---:|
|floor_reference|-0.7626 to 1.6853|14 / 5|-0.9532 to 1.9706|0.182786|0.0036 to 0.3901|
|incumbent_reference|-0.0089 to 1.1520|33 / 1|-0.0387 to 1.3677|0.000000|0.0009 to 0.3121|
|add_only|0.0074 to 1.1750|33 / 0|-0.0336 to 1.3724|0.000000|0.0008 to 0.2879|
|remove_only|-0.0461 to 0.0000|0 / 30|-0.0527 to 0.0192|0.218896|0.0000 to 0.0456|
|ridge_incumbent|-2.1309 to 2.7465|21 / 3|-0.2770 to 3.6064|5.214003|0.0263 to 0.2045|
|old_stop|0 to 0|0 / 0|0 to 0|0.000000|0 to 0|
|previous_matched|-0.9007 to 1.6471|12 / 3|-0.8374 to 1.9401|0.237668|0.0032 to 0.3837|
|raw_neural|-21.5626 to 10.8546|24 / 6|-4.5749 to 14.1204|81.295455|0.6512 to 0.9963|

| Incumbent-relative vs | All gain (%) | Positive / negative CI | Hard gain (%) | Endpoint FDE gain (%) |
|---|---:|---:|---:|---:|
|floor_reference|-0.5684 to 0.9833|19 / 8|-0.9331 to 0.9919|-0.7029 to 1.4924|
|ridge_incumbent|-2.8094 to 2.4142|13 / 18|-3.6882 to 1.2143|-4.6858 to 4.5689|
|old_stop|-0.0089 to 1.1520|33 / 1|-0.0387 to 1.3677|-0.0005 to 1.6938|
|previous_matched|-0.5618 to 1.1194|19 / 9|-0.9156 to 0.8953|-0.6540 to 1.7753|

## Learned Risk Is Not Calibrated Safety

| Head | Supported selected locality/views | Realized harm ratio >2% | Underpredicted | Realized ratio range |
|---|---:|---:|---:|---:|
|floor_reference|144|60|119|0.0000 to 0.1435|
|incumbent_reference|139|65|110|0.0000 to 0.2341|
|ridge_incumbent|144|132|139|0.0000 to 3.4402|

For incumbent-relative heads, selected means an override, and the harm denominator is the incumbent cost; for floor heads it means selecting neural over the floor.
These are different reference-cost ratios, not directly interchangeable safety certificates. Net easy degradation is a separate quantity.

Every source/event/seed, partial-label and complete-window comparison, p95/p99 tail, endpoint FDE, zero-CV result and changed-action accounting is retained in groups/*.json.
Bootstrap:3,000 paired source-locality resamples, four localities per view. Repeated windows are not IID. No multiplicity-adjusted confirmatory claim.
Same feature statistics, supervision masks, source weights, draw counts, capacity, seeds and budget are verified. Target-dependent initial biases and ranking scales legitimately differ.
No new trajectory forecaster training, independent calibration, reserved-role opening or deployment. Image pixels, obs8/pred12 rawstride12, released detector tracks; no metric/seconds/human-gold/physical-safety/true3D/foundation claim.
