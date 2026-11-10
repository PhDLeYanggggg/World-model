# One optional rotation reduces losses, without establishing a joint gain

The [exact direction menu](../opendd_vehicle_angular_pair_v1/README.md)
shows that one rotation is usually enough in hindsight. I kept
both original edited actors, every step's displacement magnitude,
and the original recording-level checks. The new rule either
keeps both paths or rotates one. It requires positive individual
and relative-motion gain against all three causal surrogate
forecasts. The matched independent control uses the same rotating
vehicle at the same query, but ranks angles by individual gain only.

| Policy | Rotating views | Dual passes / 8,874 | Risk passes |
| --- | ---: | ---: | ---: |
| Original additive | 0 | 2,975 | 4,379 |
| Individual surrogate | 952 | 2,891 | 4,336 |
| Interaction-aware surrogate | 364 | 2,962 | 4,369 |
| Individual angle on the exact interaction mask | 364 | 2,952 | 4,367 |

The direct conservative trajectory difference at the same rotating
vehicles is -0.00000246 pp (95% location interval
[-0.00002711, +0.00002405]); the conservative relative-motion
difference is -0.00008565 native units [-0.00015165, -0.00003209].
The difference is much smaller than with two forced rotations,
but it does not support an interaction gain. The interaction arm
also loses conservative trajectory utility against the unrotated
incumbent (-0.000258 pp [-0.000389, -0.000158]).

The 3,730 undefined risk views are unchanged. Counts and intervals
come from 126 directions, with three seeds and seven locations, not
8,874 independent recordings. No thresholds, training, external
scores or release decisions change.

[Protocol](protocol.md) | [Grouped readout](readout.json)

**Input correction.** The distance weights in this run came from
agent-centered histories. Their final points are all zero, so the
relative term averaged neighbors without actual spacing. The counts
above remain valid for that rule; they are not a test of observed
distance weighting. See the [registered spacing and neural-context
comparison](../opendd_vehicle_direction_context_v1/protocol.md).

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_vehicle_single_rotation.py
```
