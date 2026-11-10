# Train against the forecast that would actually be replaced

The earlier angular head raised dual-pass counts, but did not
establish a conservative gain over the incumbent. Here I kept
the 780 original recipients and trained three heads at the exact
same stepwise radii. The control optimizes the worse of CV and
OLS4 relative gain. The other heads also pay the conservative
disagreement cost against the unrotated incumbent. In the tail
arm, half of the incumbent term comes from the two worst donor
means in each group-balanced minibatch.

| Objective | Dual passes / 606 | Risk passes | Lower gain vs incumbent, pp [95% location interval] | Worst location, pp |
| --- | ---: | ---: | ---: | ---: |
| Original direction | 60 | 88 | Reference | Reference |
| Dual reference | 76 | 112 | -0.01292 [-0.04789, +0.00968] | -0.11208 |
| Add incumbent mean | 71 | 107 | -0.01191 [-0.04816, +0.01086] | -0.11662 |
| Add incumbent donor tail | 74 | 110 | -0.00872 [-0.04431, +0.01273] | -0.11142 |

The tail objective improves the point estimate and is positive in
five of seven locations. Its direct conservative comparison against
the same-run dual-reference head is +0.00180 pp
[-0.00286, +0.00703]. The worst location barely changes. The
test does not support a new deployment or a tail-utility claim.
The learned angles are not the main obstacle to direction capacity:
the [fixed-recipient oracle](../opendd_angular_incumbent_oracle_v1/README.md)
finds far more useful paths than any learned head realizes.

Native arm64 training completed 63 fits (seven held locations,
three seeds, three objectives), with 25,200 updates in 30.2 seconds.
The 7.8-second checkpoint replay reproduced the readout with zero
updates. Full scoring
uses 3,000 location draws after averaging seeds. No held outcomes
choose weights, checkpoints or thresholds. Risk undefined remains
397 views for every arm. The grouped readout includes available
trajectory errors, per-location tails and easy degradation.

[Protocol](protocol.md) | [Readout](readout.json) | [Checkpoint receipts](head_freeze.json)

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4
.venv-pytorch/bin/python scripts/run_m3w_opendd_exchange_angular_incumbent.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_exchange_angular_incumbent.py --replay
```
