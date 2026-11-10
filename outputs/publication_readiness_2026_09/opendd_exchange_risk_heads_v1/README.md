# Does explicit risk supervision improve the exchange choice?

The [exhaustive audit](../opendd_exchange_attrition_v1/README.md) found
37 risk-related failures among 39 wrong choices on rescuable views.
I trained two equally initialized models on the same donor inputs,
samples and targets. Both learn the original transition and paired
gain outputs. One also learns whether risk is measurable, whether
measurable risk passes, and whether a passing exchange beats both
utility references. Their product estimates the chance of passing
all three stages. Each ranking must choose on the same 101 views
selected by the frozen aligned-context model.

| Ranking | Risk undefined | Risk violation | Risk and both utilities pass | Rescued | Destroyed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen, no exchange | 397 | 121 | 60 | 0 | 0 |
| Frozen aligned-context ranker | 399 | 128 | 47 | 5 | 18 |
| Equal-architecture control | 401 | 124 | 48 | 7 | 19 |
| With risk supervision, original ranking | 397 | 125 | 50 | 5 | 15 |
| With risk supervision, factored ranking | 397 | 130 | 49 | 7 | 18 |
| Uniform legal exchange, exact expectation | 397.33 | 124.87 | 51.21 | 6.76 | 15.55 |

All arms retain 780 edits over 606 recording-seed views, the same
movement caps, and the original 2% rule. The paired equal-location
dual-pass difference for factored ranking versus the equal-architecture
control is -0.031 pp (95% interval [-0.317, +0.225]); versus random it
is -0.557 pp ([-1.484, -0.026]). These intervals resample locations
3,000 times after averaging seeds, so they need not have the same
sign as the unweighted difference in counts. Factoring the targets
reduces undefined cases versus the control but produces more measured
risk violations. It does not improve the decision enough to use it.

The native arm64 run trained 42 heads across seven held locations and
three seeds: 16,800 updates in 74.7 seconds. Replaying the same
checkpoints took 14.4 seconds and used zero optimizer updates. Neither
the fixed intervention mask nor thresholds were refit. No new
external outcomes were used; this is a development ablation.

[Protocol](protocol.md) | [Registration](registration.json) |
[Grouped readout](readout.json) | [Checkpoint receipt](head_freeze.json)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_exchange_risk_heads.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_exchange_risk_heads.py --replay
```

The first command resumes from a local checkpoint every 100 updates.
The second requires completed checkpoints and cannot update weights.
Both use four Torch threads, zero data workers, and a PID heartbeat.
