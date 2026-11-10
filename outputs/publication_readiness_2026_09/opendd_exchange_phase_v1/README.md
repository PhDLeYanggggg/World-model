# Do earlier or later deviations improve an otherwise identical exchange?

Each alternative changes only the timing of the incoming trajectory's
deviation from CV. The recipient, original maximum deviation, step
of the peak, and final point are fixed. Interpolating between existing
residuals cannot increase the maximum norm. The alternative is
constructed without target positions; the endpoint/FDE cannot change.

| One-exchange menu, plus keeping the incumbent | Positive dual passes | Gain > 0.10 pp | Gain > 0.50 pp | Deletion-stable rescues |
| --- | ---: | ---: | ---: | ---: |
| Original | 181 | 68 | 4 | 93 |
| Early | 175 | 72 | 6 | 91 |
| Late | 178 | 62 | 3 | 91 |
| Three shapes | 182 | 73 | 6 | 97 |

These are exact hindsight counts on the same 606 views and 7,198
legal exchanges, not a learned policy. The three-shape menu barely
expands the positive frontier. It adds five views above 0.10 pp,
and the total above 0.50 pp remains six.

At the same 101 exchanges already chosen by the frozen ranker,
late timing raises dual passes from 47 to 51 and risk passes from
79 to 86. The paired conservative utility difference over the
original shape is +0.000435 pp (95% location interval
[+0.000056, +0.001044]). Early timing gives 47 passes and
-0.000676 pp [-0.001700, -0.000077]. The first-three-step OLS4
control chooses early in 100 of 101 views and does not improve
the result. The fixed incumbent without swaps still reaches 60.

Peak matching alone does not keep the sum of per-step deviations.
The [cumulative-deviation control](../opendd_exchange_matched_amplitude_v1/README.md)
separates timing from this change in how much the forecast departs
from CV over the whole path.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_phase.py
```
