# OpenDD: can the model pick the right edit exchange?

I tested one exchange without changing the forecast menu, 780 realized
edits, per-recording movement caps, time credits or original 2% risk rule.
The ranker sees the observed-history features of the two affected queries.
The contextual arm also sees a pooled summary of the original allocation.
Both arms train on exactly the same donor examples; the scored location
is excluded. The full score sequence and count are available for this
matched comparison, so it is offline, not a streaming release policy.

| Rule | Exchanged / 606 | Rescues | Destroys | Risk and both references / 606 |
| --- | ---: | ---: | ---: | ---: |
| Frozen, no exchange | 0 | 0 | 0 | 60 |
| Minimum movement | 294 | 30 | 39 | 51 |
| Local guarded | 179 | 11 | 16 | 55 |
| Contextual guarded (primary) | 113 | 10 | 15 | 55 |
| Contextual, no gain guard | 150 | 15 | 18 | 57 |
| Contextual, forced best score | 296 | 30 | 30 | 60 |

The primary arm's paired conservative gain versus no exchange is
-0.00443 percentage points (95% location interval [-0.00882, -0.00066]).
Pooled context changes intervention frequency, but does not recover more
dual passes than the local model. The forced choice cancels its own
rescues with destruction of previously passing plans.

## Exhaustive and same-view random controls

I scored all 7,198 legal exchanges, and kept the anchor as another option.
This resolves the earlier one-exchange feasibility ranges without an
optimizer: **181** views can have positive gain, **68** exceed 0.10 pp,
**4** exceed 0.50 pp and **0** exceed 1.00 pp. Of the 121 newly confirmed
positive opportunities, the guarded model recovers 10. The earlier
178--183 count was a valid bound, not a different result.

The random control acts on the exact same views as each model and uses the
uniform mean over all their legal exchanges. It gives 53.51 expected dual
passes for the guarded intervention mask and 52.59 for the forced mask.
The paired equal-location differences (model minus random) are -0.32 pp
[-1.40, +0.39] guarded and +0.29 pp [-1.37, +2.13] forced. These are
3,000 location-bootstrap intervals after averaging seeds. Neither is a
reliable improvement over chance, even though the pooled forced count
is higher. The exact average avoids Monte Carlo random-control noise.

## Reproduction and status

Seven locations by three seeds, 42 heads, 16,800 updates. Native arm64
training finished in 60.3 seconds. Checkpoint replay took 4.9 seconds with
zero updates. The 67,275 donor training examples across folds are not
independent recordings. No held threshold search or external outcomes.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/run_m3w_opendd_one_exchange_ranker.py --replay
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_one_exchange_ranking.py
```

Omit `--replay` to fit or resume. The [protocol](protocol.md),
[registration](registration.json), [checkpoint manifest](head_freeze.json),
[readout](readout.json) and [ranking diagnostic](ranking_diagnostic.json)
bind the numbers. The existing policy stays in place.
