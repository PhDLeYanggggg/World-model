# Hindcast-Anchored Forecast Training

**Development result: better than causal hindcast, not a safe replacement.**
The easy-harm arm passes the registered utility comparison against CV and
matched hindcast. It fails recording-level safety. The post-hoc displacement
control shows that a larger edit explains the apparent improvement over
the incumbent neural forecast at the same 5% coefficient.

[Frozen protocol](protocol.md) | [Location/seed metrics](analysis.json) |
[Displacement audit](amplitude_audit.json) | [Manuscript supplement](../evidence_manuscript_v30/README.md)

## Fixed 5% comparison

63 fresh native Torch fits, three objectives, seven held OpenDD locations,
and three seeds. Each fit excluded its held location and used 600 updates.
The models start at the emitted 5% hindcast blend, and learn bounded local
velocity increments. Every edited arm makes the same 30,075 target edits
across the three views of the 10,475-target cohort. No threshold was chosen.

| Arm | Lower gain vs CV (pp) | vs matched hindcast (pp) | Original risk passes |
| --- | ---: | ---: | ---: |
| Hindcast | -0.117 [-0.322, +0.035] | Reference | 399/606 |
| Incumbent neural blend | +0.429 [+0.282, +0.643] | +0.375 [+0.207, +0.582] | 455/606 |
| Observed-only training | +1.391 [-0.521, +3.721] | +1.419 [-0.460, +3.722] | 16/606 |
| Add missing-step debit | +1.731 [+0.146, +3.788] | +1.758 [+0.244, +3.742] | 26/606 |
| Add positive easy harm | +1.279 [+0.445, +2.185] | +1.313 [+0.598, +2.130] | 52/606 |

Intervals use 3,000 paired location draws after averaging the three seeds.
The prespecified easy-harm arm is positive against hindcast at seven of
seven locations, including complete rows (+1.659 pp [0.939, 2.402]).
The positive-harm penalty improves the recording pass count from 26 to 52
at fixed recipients, but does not make the policy safe. Thirty views lack
full easy reference. The worst defined signed easy degradation is 103.99%;
small recording denominators can make this ratio large, and those failures
remain failures. Neither the model nor the policy is deployed.

## Post-hoc realized displacement controls

The new arm moves farther from CV than the incumbent in 29,092 of 30,075
matched edits. Median per-view displacement ratios range from 2.46 to
9.92. Equal blend coefficients therefore do not imply equal edit size.

| Restriction for easy-harm model | Lower gain vs CV | vs matched hindcast | vs incumbent | Risk passes |
| --- | ---: | ---: | ---: | ---: |
| At most incumbent radius | +0.310 | +0.198 | -0.166 [-0.268, -0.080] | 473/606 |
| Same realized radius for all edited arms | +0.208 | +0.094 [+0.011, +0.164] | -0.116 [-0.215, -0.030] | 527/606 |

The radius is the maximum per-step distance from CV over the emitted path.
The equal-radius comparison shrinks every arm to the smallest existing
radius for that target, without changing the direction in path space.
These are outcome-blind transformations, but were tested after seeing
the primary failure; they do not select a threshold or model. The original
gate still fails, including undefined easy-reference cases.

The next experiment should place the incumbent-derived radius inside
the decoder and loss, rather than shrink after training. The relevant
contrast is shape utility against the incumbent under an equal realized
budget, followed by unchanged per-recording risk. The two comparisons
separate motion representation from intervention size.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_hindcast_residual.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_hindcast_residual.py run
.venv-pytorch/bin/python scripts/run_m3w_opendd_hindcast_residual.py replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_hindcast_amplitude.py
~~~

Training took 24.3 seconds locally with four Torch threads and no workers;
checkpoints include optimizer and draw state every 100 updates. Replay
loaded 63 hash-bound checkpoints, made no updates and matched the complete
readout. The displacement audit also matched on rerun. 19 targeted tests
passed. No CREATE job was needed. TUMDOT was not rescored and the 30
DroneCrowd confirmation groups were not accessed. No deployment, metric or
seconds claim for SDD, true 3D, foundation claim, Stage5C, or SMC.
