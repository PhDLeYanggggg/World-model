# What changes when every eligible view swaps?

The [matched 101-view comparison](../opendd_exchange_stability_head_v1/README.md)
is concentrated at one location. I retained that analysis and froze
the trained models. For this separate test each ranker must choose
one exchange on all 296 views with a legal option. This expands coverage
from four to six locations without changing the 780 original edits,
movement caps or original 2% risk and utility checks. The seventh
location has no legal exchanges. The largest location still supplies
200 of 296 swaps, so equal-location paired intervals matter.

| Ranking | Original dual passes / 606 | Stable rescues | Rescued | Destroyed |
| --- | ---: | ---: | ---: | ---: |
| Frozen, no swaps | 60 | 0 | 0 | 0 |
| Previous aligned-context ranker | 53 | 22 | 26 | 33 |
| Equal-architecture risk control | 59 | 21 | 28 | 29 |
| Auxiliary model, original full-pass score | 53 | 22 | 27 | 34 |
| Auxiliary model, stability-weighted score | 61 | 28 | 32 | 31 |
| Exact same-view uniform expectation | 52.59 | 17.69 | 24.83 | 32.24 |

Stability weighting adds two raw dual passes and seven stable rescues
over the architecture-matched control, but the corresponding paired
equal-location intervals are [-2.102, +0.300] pp and [-1.829, +0.873] pp.
The dual-pass interval against uniform choice is [-1.353, +2.170] pp,
and against no swaps [-2.540, +3.969] pp. These 3,000-draw intervals
average seeds within locations. The larger mask shows the earlier
concentration was not the only obstacle. The outcome is a diagnostic
signal for candidate quality, not a reproducible gain or a release.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_stability_forced.py
```
