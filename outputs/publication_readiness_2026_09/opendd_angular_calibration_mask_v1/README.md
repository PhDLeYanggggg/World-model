# The fixed angle accounts for most of the calibrated gain

All 198 rotations are held fixed for the direct angle comparison.
Only the angle changes. The simple angle is selected using donor
data only and remains constant within each held site and seed.

| Paired contrast | Conservative utility, pp | 95% location interval |
| --- | ---: | ---: |
| Donor constant minus learned angle, same mask | +0.007378 | [+0.000224, +0.018790] |
| Donor constant versus incumbent | +0.006138 | [+0.000664, +0.015826] |
| Donor constant on an exact same-count random mask versus incumbent | +0.005833 | [+0.000316, +0.015623] |
| Learned mask minus same-count random mask, both with donor constant | +0.000305 | [+0.000016, +0.000697] |
| Uniform nonzero angle on learned mask versus incumbent | -0.003264 | [-0.009666, +0.001595] |

Uniform placement holds each recording-seed view's rotation
count fixed. Only 44 of the 114 rotating views permit a different
placement. For the additive conservative contrast versus the
incumbent, the expectation is exact, with marginal probability
`k/n` per eligible row. The mask comparison is a difference of
utilities versus the same incumbent, not a direct path-to-path
bound. Risk-pass counts are not inferred from these expectations.

This supports a small benefit from the donor-constant angle,
not a strong learned recipient or angle policy. Almost all of
the positive point estimate survives randomizing the mask at the
same recording counts. The development result does not change
the release rule or the paper's independent conclusions.

[Protocol](protocol.md) | [Grouped readout](readout.json)

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/audit_m3w_opendd_angular_calibration_mask.py
```
