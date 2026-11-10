# Changing amplitude recovers the apparent late-timing gain

The late warp's cumulative deviation is attainable by rescaling
the original, non-warped path at all 101 chosen exchanges. The
control retains each step's residual direction, peak steps and
endpoint. It matches both the peak radius and the sum of per-step
deviations to the late path. No changed recipients, edited-label
support or future input.

| Identical chosen exchanges | Dual passes / 606 | Risk passes / 606 |
| --- | ---: | ---: |
| Original shape | 47 | 79 |
| Late phase warp | 51 | 86 |
| Non-warped amplitude control | 52 | 85 |
| Frozen incumbent without swaps (reference) | 60 | 88 |

| Paired conservative effect | pp | 95% location interval |
| --- | ---: | ---: |
| Late minus original | +0.000435 | [+0.000056, +0.001044] |
| Amplitude minus original | +0.000393 | [+0.000059, +0.000916] |
| Late minus matched amplitude | -0.000021 | [-0.000060, +0.000003] |

The gain of the late warp over the original is small and a simple
amplitude control recovers it. The direct comparison does not
establish an added effect of the time warp. The independent
[equal-total redistribution](../opendd_exchange_matched_amplitude_v1/README.md)
also finds no late timing gain. This is a matched development
result, not a new policy. Endpoints and the original 2% rule are
unchanged. No external outcomes were opened.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_amplitude_control.py
```
