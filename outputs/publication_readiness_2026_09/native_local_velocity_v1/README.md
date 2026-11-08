# Do nearby trajectories improve velocity forecasts?

I replaced the fixed position-anchor edit with a learned velocity
adjustment. The model keeps the observed position and predicts a
short-lived and a sustained velocity change relative to OLS4.
It sees eight annotation steps and outputs twelve. I compare it with
the same model using either the outer half of the same nearest-eight
bank or no peers at all. The two peer arms see equal numbers of
complete-history neighbors, and all three use identical training draws
per fold. The outer control is not a claim that its agents never
interact. [Registered protocol](protocol.md)

## What changed, and what did not

Fresh native Torch training: 81 fits, 800 updates each, seeds 17, 29
and 43, nine leave-one-fitting-source-out folds. The original 69,746
windows belong to 1,176 queries; 67,854 have any future label, but only
45,330 have all twelve. Intervals resample the nine sources 3,000 times
after seed averaging. The three columns use the same CV error-mass
denominator; they are not percent improvements in ADE or FDE.

| Contrast | Available Labels | Wholly Unknown Rows Debited | All Missing Steps Debited [95% CI] |
|---|---:|---:|---:|
| Near vs OLS4 | +0.0263 | +0.0173 | -0.0184 [-0.0294, -0.0071] |
| Near vs ego | +0.0117 | +0.0054 | -0.0195 [-0.0273, -0.0107] |
| Near vs outer | -0.0077 | -0.0198 | -0.0602 [-0.1020, -0.0222] |
| Fixed near rule vs OLS4 | +2.2415 | +1.7380 | -1.0209 [-2.4242, +0.2554] |
| Near vs CV | +0.6040 | -10.9461 | -38.4711 [-66.0398, -18.5033] |

For each missing step, the difference between forecast and comparator
positions bounds the worst possible loss in Euclidean error. The last
column applies this bound at each missing step; the middle column
retains the historical available-label estimate on partially labeled
rows. A negative lower bound does not identify what the missing
positions actually were. It means the available data do not establish
a positive gain under that bound.

The median change at the final step for the near arm is only
0.0183 to 0.0274 image pixels across seeds. The network barely moves
from its OLS4 initialization. The conservative, harm-weighted loss is
a plausible reason, but this experiment does not isolate that cause.
On available labels, near ADE/FDE are 91.567/234.418 image pixels
versus 91.597/234.452 for OLS4. One easy source has no denominator in
each seed; the worst supported positive easy-harm ratio exceeds 260%
in all near seeds. That is a diagnostic of unrestricted edits, not a
formal risk evaluation. There is no easy-safety claim.

## Does shortening the correction help?

The fixed rule is larger than the learned adjustment and has a larger
available-label gain. I used one follow-up schedule, fixed before its
readout: preserve the rule through step 6, then smoothly return to
OLS4 by step 12. It leaves FDE unchanged from OLS4 by construction.
Its available-label score is +1.9260 points; the full-grid lower
estimate is -0.2308 [-0.9376, +0.4560]. This is less negative than the
unshortened rule's lower estimate, but not evidence that the pulse
improves on it: their direct paired lower contrast is -1.3930
[-2.2609, -0.7086]. Near vs outer on 69,615 jointly nonidentity rows
also fails: -0.5144 [-0.8381, -0.2523]. This was a follow-up on the
same fitting sources, not an independent experiment.
[Diagnostic design](pulse_protocol.md)

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_local_velocity.py train
.venv-pytorch/bin/python scripts/run_m3w_local_velocity.py score
.venv-pytorch/bin/python scripts/run_m3w_local_velocity.py replay
.venv-pytorch/bin/python scripts/report_m3w_local_velocity_pulse.py
```

Use `train --resume` to continue saved state. Native arm64 CPU used
four compute threads and no loader subprocesses; checkpoints were
written every 100 updates. The first fold's completed checkpoints
were reused after correcting a two-argument validation helper in the
three-arm driver. The [amended registration](registration_v2.json)
binds the original checkpoint identity. Nine focused tests passed.
The [main results](analysis.json) reproduced byte-for-byte on replay,
and the [pulse diagnostic](pulse_diagnostic.json) reproduced on a
second run. Private arrays and weights are not included.

The results do not support a near-neighbor neural forecast advantage.
The original 2% safety rule and the DroneCrowd 15/30 split remain
unchanged. No development, external calibration or independent
confirmation outcomes were used. These image coordinates and native
steps do not imply a metric, seconds, true 3D, or foundation model.
No deployment, Stage5C, or SMC.
