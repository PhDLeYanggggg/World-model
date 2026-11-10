# Athens: past-only inputs and matched proposals

I turned the audited pNEUMA pilot into model inputs before opening
any prediction targets. The engineering run uses the ten-tick
8/12 windows already measured in the [source audit](../pneuma_source_intake_v1/README.md).
It does not register that stride as the formal experiment clock or
assign any date to calibration or confirmation.

The first pass uses track identity, timestamps and the presence of
historical position tokens to reconstruct the original rosters. The
second parses numeric positions only at the eight selected history
ticks. It reproduced 643 windows and 67,962 vehicle-window instances
across the six CSVs. There is no future-coordinate or future-index
array in the input packets. A synthetic test replaces numeric
future coordinates with opaque text or blanks and produces the same
past packet.

I use a fixed horizontal tangent plane from WGS84, with origin
38 degrees north, 23.75 degrees east and zero ellipsoidal height.
This uses the [WGS84 ellipsoid](https://earth-info.nga.mil/index.php?action=wgs84&dir=wgs84)
and [topocentric transform](https://proj.org/en/stable/operations/conversions/topocentric.html).
I translate each query to its first roster member's last observation,
without fitting the origin, scale, or a normalizer to target outcomes.
The resulting plane is in metres by construction. That does not
verify the accuracy of the released trajectories, and it says
nothing about the unverified SDD geometry.

## Frozen inference

The same input histories run through CV, OLS4, median velocity and
the existing OpenDD vehicle completion forecaster (21 verified
checkpoints, seven per seed). No new weights are fitted. All 643
windows have both uniform-family and unrestricted two-edit proposals
at the original relative radius for each of the three seeds.

The largest input-only roster has 241 vehicles. In the engineering
window with the most legal choices for seed 17, 237 vehicles yield
141,802 unordered pairs of distinct recipients. Streaming the
search in at most 4,096 pairs per block reproduces the four original
argmax choices and scores exactly, including tie order. The largest
relative-feature array for this window shrinks from about 3.40 MB
to 0.10 MB; this is not a claim about total process memory.

The four arms are independent ranking, additive pair ranking,
same-family ranking, and unrestricted joint ranking. Each proposes
two distinct vehicle trajectories at the same peak displacement
from CV within each seed and query. The full bank covers all 1,929
query-seed views in 18 independently resumable packets. The joint
proposal differs from independent, additive and same-family choices
in 1,759, 1,730 and 130 views, respectively. This is proposal
variation, not utility. No Athens threshold, risk certification or
forecast accuracy result exists.

## Reproduce the unscored pilot

Run these from the repository root in the native arm64 Torch environment.
Existing donor weights and downloaded pilot members are required.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/build_m3w_pneuma_prefix_pilot.py prefix
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/build_m3w_pneuma_prefix_pilot.py forecast
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_pneuma_pilot_slots.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/benchmark_m3w_pneuma_pair_memory.py
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/build_m3w_pneuma_matched_pilot.py
```

The builders verify hashes and resume from completed per-file or
file-seed packets. The heartbeats and arrays remain local. The
future join and sparse pair bounds have synthetic tests only;
no part of this run opened the actual future positions or ran
CREATE. The [prefix](prefix_receipt.json), [forecast](forecast_receipt.json),
[slot](slot_receipt.json), [exact ranking](pair_stream_receipt.json)
and [matched proposal](matched_receipt.json) receipts are light and
can be checked without the raw files. The independent test still
requires the approved date roles and frozen risk/readout contract.
