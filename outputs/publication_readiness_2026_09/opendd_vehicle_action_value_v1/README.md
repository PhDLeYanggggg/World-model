# OpenDD: learn the gain from a change, not another forecast

The previous learned forecasts reduced ADE but did not choose useful
rotations. Here I trained the same network twice, with and without
neighbor and partner inputs, to predict the gain and harm of each
legal rotation relative to the **exact original direction**. Missing
steps contribute no observed gain and are charged their maximum
possible loss. The two original actors, prediction query and per-step
displacements from CV are fixed.

## What the models picked

| Rule | Rotations | Risk passes | Risk and both references |
| --- | ---: | ---: | ---: |
| Original, no rotation | 0 | 4,379 | 2,975 |
| Ego-only action value | 526 | 4,406 | 3,001 |
| Context action value | 2,399 | 4,359 | 2,908 |
| Context with relative-gain gate | 167 | 4,380 | 2,980 |
| Ego angle, same actors as context | 2,399 | 4,391 | 2,980 |
| Donor-selected constant, same actors | 2,399 | 4,201 | 2,615 |
| Relative-gain term, same actors | 2,399 | 4,373 | 2,941 |

The 8,874 repeated views cover 493 recordings; 3,730 undefined
risk views remain undefined. Location bootstrap intervals use 3,000
paired draws after averaging directions and seeds.

| Paired contrast | Conservative trajectory gain (pp) | Conservative relative gain (native units) |
| --- | ---: | ---: |
| Ego minus no rotation | +0.000088 [-0.000001, +0.000232] | -0.000657 [-0.001313, -0.000180] |
| Context minus no rotation | -0.001005 [-0.001750, -0.000386] | -0.006235 [-0.010691, -0.002502] |
| Context minus constant, same actors | +0.002394 [+0.001298, +0.003840] | -0.003264 [-0.006736, -0.000231] |
| Context minus ego, same actors | -0.000822 [-0.001495, -0.000271] | -0.004773 [-0.008650, -0.001724] |
| Add relative-gain term, same actors and model | +0.000398 [+0.000146, +0.000684] | -0.000909 [-0.002011, -0.000062] |

Learning action value beats the fixed angle on the same actors.
That does not beat the ego model or the unrotated incumbent.
Adding a relative objective improves the trajectory contrast against
the same model without that term, but worsens the relative-motion
metric it was meant to improve. None of this changes the release.

## What calibration and hindsight show

Each arm finds a nonempty calibration prefix in 16/126 directions.
The frozen context rule rotates 18 held views: 13 pass the original
risk check and 10 also beat both references. The joint rule rotates
one view; it passes risk but not both references. Held support is
not a certification, and the other views are unchanged rather than
new successes.

An independent finite-menu replay recovers the previous exact
**1,393** views with a legal single rotation that passes both
references and risk, preserves conservative trajectory utility
against the original, and improves conservative relative motion.
Among those 1,393 views the context rule misses **855** by not
rotating, **96** by choosing the wrong actor, and **281** by choosing
the wrong angle. It captures **161**, versus **99** for the
fixed angle and **169** for the ego angle at the same recipients.
The unrestricted joint rule captures **25**, with 1,324 abstentions
on the 1,393 eligible views. These exact counts diagnose the
decision hierarchy. They are hindsight, not reachable performance
without a learned gate.

The next useful comparison is a source-held predictor of whether
any worthwhile intervention exists, before choosing the actor or
angle. This is more directly motivated than widening the angle menu.
The existing threshold and external confirmation stay unchanged.

## Replay

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_vehicle_action_value.py freeze
.venv-pytorch/bin/python scripts/run_m3w_vehicle_action_value.py score
.venv-pytorch/bin/python scripts/audit_m3w_vehicle_action_capture.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_vehicle_action_value.py
```

`fresh_run`: 126 native arm64 Torch fits, 25,200 updates, all
three seeds, plus the frozen evaluation and capture diagnostic.
The pre-score freeze was pushed as `052c7b3a2`. The 141-test
regression suite includes exact utility labels, resume and mask
checks. `cached_verified`: observed geometry, fixed recipients,
cohorts and source-excluded forecasts. The sampled donor actions
are not the actual incumbent policy's distribution. OpenDD is
exposed development, not a new independent or metric claim.
No external outcomes were opened.

[Protocol](protocol.md) | [Freeze](freeze.json) | [Paired readout](readout.json) |
[Exact capture readout](capture_readout.json).
