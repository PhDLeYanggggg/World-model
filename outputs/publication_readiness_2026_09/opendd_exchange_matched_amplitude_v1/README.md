# When total deviation is fixed, moving it later does not help

The [first timing comparison](../opendd_exchange_phase_v1/README.md)
matches peak displacement and endpoint, but the sum of per-step
deviations can change. This control moves norm mass between the first
and second halves of the forecast while also fixing that sum and
retaining each step's residual direction. Both alternative profiles
change all 7,198 legal receiving paths. No future position enters
their construction.

| One-exchange menu, plus keeping the incumbent | Positive dual passes | Gain > 0.10 pp | Gain > 0.50 pp | Deletion-stable rescues |
| --- | ---: | ---: | ---: | ---: |
| Original | 181 | 68 | 4 | 93 |
| Front-loaded | 171 | 63 | 4 | 84 |
| Back-loaded | 181 | 69 | 4 | 93 |
| Three shapes | 182 | 71 | 4 | 95 |

The expanded exact frontier changes by one view, and its
0.50-pp frontier does not change. At the same 101 frozen
exchanges the back-loaded arm gives 47/606 dual passes, equal to
the original. Its paired conservative gain is -0.000012 pp
with 95% location interval [-0.000034, +0.000002]. Front-loading
gives 50 dual passes but decreases paired utility by -0.000234 pp
[-0.000659, -0.000007]. In this experiment changing timing at
matched total deviation does not improve both checks.

[Protocol](protocol.md) | [Grouped readout](readout.json) |
[Direct amplitude comparison](../opendd_exchange_amplitude_control_v1/README.md)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_matched_amplitude.py
```
