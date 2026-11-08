# Forecast Shape at a Fixed Realized Budget

**Result: better forecast shape, worse recording safety.** Training at
the old model's realized displacement radius recovers a small improvement
over that model on every OpenDD development location. It does not
qualify the new model for deployment or open independent confirmation.

[Registered protocol](protocol.md) | [Frozen results](analysis.json) |
[Recording harm attribution](risk_attribution.json) |
[Supplement](../evidence_manuscript_v31/README.md)

## Equal-radius results

84 fresh Torch fits (four objectives/input arms, seven held locations,
three seeds), 50,400 updates. All edited arms use the same 30,075
target-seed edits and match the incumbent's maximum departure from CV
per target over 12 steps. The incumbent's training-side shapes come from
63 verified pair-excluded producers, not models that saw the fitted or
held location. Twenty-one held-side producers each exclude their held
location. Only previously exposed OpenDD data were used.

| Arm | Lower gain vs CV (pp) | Lower gain vs incumbent (pp), 95% CI | Original risk passes |
| --- | ---: | ---: | ---: |
| Matched-radius hindcast direction | +0.002 | -0.526 [-0.611, -0.449] | 416/606 |
| Incumbent | +0.429 | Reference | 455/606 |
| Observed error | +0.511 | +0.068 [+0.029, +0.124] | 415/606 |
| Add missing-step debit | +0.510 | +0.068 [+0.034, +0.117] | 417/606 |
| Add positive easy harm (primary) | +0.498 | +0.058 [+0.025, +0.102] | 425/606 |
| Add pedestrian history to primary | +0.491 | +0.045 [-0.008, +0.111] | 428/606 |

The primary arm beats the incumbent at all seven locations, including
on complete trajectories (+0.075 pp [0.041, 0.114]), and beats the
matched hindcast direction (+0.395 pp [0.301, 0.542]). The
paired 3,000-draw location bootstrap averages seeds before resampling;
these seven locations are development evidence, not seven independent
confirmation trials. The primary result differs from the previous
post-hoc amplitude study: equal radius was enforced in training here.
This is not evidence of external transport.

Adding neighbors to the otherwise identical primary model gives
-0.025 pp [-0.058, +0.007] in the direct paired utility contrast.
The mean relative-vector error difference is +0.000011 in release
coordinates [-0.000181, +0.000183]; smaller is better. The apparent
three additional recording passes do not establish an interaction
forecast gain or certify physical safety.

## Why safety worsened

At equal recipients and maximum departure from CV, both models share
the same selected easy-reference mass and the same debit for wholly
unobserved targets. Their risk difference therefore comes from
positive harm and signed degradation on observed labels. The audit
replayed both model outputs and the original rule:

| Transition from incumbent to primary | Recording-seed views |
| --- | ---: |
| Both pass | 421 |
| Lose pass | 34 |
| Gain pass | 4 |
| Both fail with defined risk | 117 |
| Both undefined | 30 |

All 34 lost passes cross the upper easy-risk limit. Four also cross
the net easy limit, and one crosses the all-risk limit (overlapping).
The primary arm's worst defined signed easy degradation is 25.08%;
the incumbent's is 18.44%. Neither meets the original complete rule.
The current pointwise penalty improves pass counts over the two
other new self-only losses but still trades easy harm for average gain.

The most direct next test is a recording-normalized positive-harm loss
with an explicit incumbent-harm comparison, keeping the exact radius
and intervention support. This targets the observed cause without
moving thresholds or treating undefined risk as zero. Continue to
compare joint or neighbor mechanisms only at a matched budget.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_radius_shape.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_radius_shape.py run
.venv-pytorch/bin/python scripts/run_m3w_opendd_radius_shape.py replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_radius_risk.py
~~~

Native arm64 CPU, four Torch threads, zero workers, 38.2 seconds for
training and readout. Resume keeps optimizer state and sampling order
at 100-update checkpoints. Zero-update exact replay and a repeated
risk audit matched; 24 targeted tests passed. No CREATE job was needed.
No TUMDOT rescoring, DroneCrowd confirmation access, new deployment,
SDD metric or seconds claim, true 3D, foundation claim, Stage5C, or SMC.
