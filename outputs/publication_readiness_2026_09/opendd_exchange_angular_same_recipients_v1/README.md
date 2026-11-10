# Apply the learned direction without changing recipients

I replayed the two trained heads on the same original 780 edits.
No edit moves to a different recipient or query, and each forecast
step stays the same distance from CV. The 397 undefined risk views
are unchanged, as expected when selected label support is unchanged.

| Direction | Dual passes / 606 | Risk passes | Paired conservative utility, pp [95% location interval] |
| --- | ---: | ---: | ---: |
| Original | 60 | 88 | Reference |
| Bounded acceleration | 57 | 75 | -0.08996 [-0.16359, -0.03887] |
| Decaying turn | 65 | 91 | -0.01878 [-0.05602, +0.00499] |
| Observed-suffix choice | 76 | 93 | -0.03190 [-0.07371, -0.00173] |
| Learned, observed error | 77 | 112 | -0.01305 [-0.05068, +0.01306] |
| Learned, dual-reference loss | 77 | 113 | -0.01272 [-0.04951, +0.01299] |

With the dual-reference head, the location-weighted gain in
dual-pass rate is +8.43 pp [+0.45, +22.71]. It comes from four
locations; the conservative utility loss is largest at the
location with 15 repeated recording-seed views. That location
remains in the metric. The available step and endpoint means
decrease slightly, but the conservative utility interval still
does not establish an improvement. This isolates a real change
in the number of qualifying views without evidence for deployment.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_exchange_angular_same_recipients.py
```
