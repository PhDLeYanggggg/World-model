# Athens: exercise the real packet sizes without opening outcomes

I ran the actual evaluation entry points using the six frozen
forecast/roster inputs and fabricated future displacements. Unlike the
small prior fixture, this checks the real 643 queries, three forecast
seeds, matched edit indices and fixed radii. The fabricated trajectories
and missing masks are produced in memory from past input and row index.
The rehearsal never calls the real future reader.

The four-date route exercises six confirmation members with no target
calibration and uses the verified donor-only cuts. The chronological
route exercises four calibration members on the first two dates, then
two confirmation members on the other dates. Both use the original 2%
rule, reject unintended date access, and reconstruct a missing summary
byte for byte from saved member receipts without another target call. The mock
publication used here does not publish or assign any real roles.

The [receipt](receipt.json) records only which paths completed. It
deliberately omits accuracy and risk estimates on artificial labels.
No formal rule, threshold or release decision was changed. Run:

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/rehearse_m3w_pneuma_real_packets.py
```
