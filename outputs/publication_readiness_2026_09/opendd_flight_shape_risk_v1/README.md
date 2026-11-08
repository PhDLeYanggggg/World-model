# Recording-Risk Training at Matched Forecast Radius

**Result: more passing recordings, less forecast utility.** Directly
training the existing recording risk improves the pass count without
changing recipients or displacement. It does not recover both safety
and trajectory gain, so none of these heads replaces the incumbent.

[Frozen protocol](protocol.md) | [Full paired metrics](analysis.json) |
[Hindsight feasibility](fixed_radius_feasibility.json) |
[Manuscript supplement](../evidence_manuscript_v32/README.md)

84 fresh Torch fits, seven held OpenDD development locations and three
seeds. The four new arms share 600 updates, initialization, complete
recording draws and 30,075 evaluated target-seed edits. Train-side
incumbent forecasts remain pair-excluded; the held location does not
set the easy/hard cuts or train the shape. The previous model and
all incumbent results reproduce exactly from their frozen weights.

| Arm | Lower gain vs CV (pp) | Lower gain vs incumbent (pp), 95% CI | Original risk passes |
| --- | ---: | ---: | ---: |
| Incumbent | +0.429 | Reference | 455/606 |
| Prior row-sampled easy-harm model | +0.498 | +0.058 [+0.025, +0.102] | 425/606 |
| Pointwise loss, recording batches | +0.465 | +0.024 [-0.006, +0.054] | 438/606 |
| Add original 2% budget penalty | +0.367 | -0.086 [-0.124, -0.060] | 465/606 |
| Add relative recording harm (primary) | +0.366 | -0.082 [-0.121, -0.051] | 476/606 |
| Add non-canceling target harm | +0.375 | -0.073 [-0.104, -0.045] | 480/606 |

Intervals are paired over locations (3,000 draws after averaging
the three seeds). Against the matched recording-batch control, the
primary model loses -0.137 pp [-0.174, -0.092]. It recovers 27 of the
incumbent's failures and loses six of its passes. The non-canceling
arm recovers 29 and loses four, but has a negative utility contrast
at all seven held locations. Its worst defined signed easy
degradation is 13.17%, still beyond 2%. The same 30 views are
undefined. The prespecified recovery criterion fails, and no
external data were accessed.

This uses a fixed training penalty, not the independently validated
constraint optimization considered by
[Cotter et al.](https://proceedings.mlr.press/v97/cotter19b.html),
nor the expected monotone-loss guarantee of
[Conformal Risk Control](https://proceedings.iclr.cc/paper_files/paper/2024/file/f3549ef9b5ff520a7e41ff3cc306ab2b-Paper-Conference.pdf).

## What the fixed budget makes impossible

As a separate post-hoc calculation, allow an arbitrary 2D offset at
each future step and use the true future. Keep the original radius
and recipients. Project each known future residual onto the
radius disk. If all steps are known and none touches the disk
boundary, enforce the radius at the least costly step, the step
with the largest true residual. With an unobserved step, it can
serve as the radius witness at no observed cost. This minimizes
observed ADE and hence all three original risk ratios together.
It omits smoothness and is not a causal prediction.

| Optimistic classification | Views |
| --- | ---: |
| Undefined easy reference | 30 |
| Wholly unobserved debit alone exceeds budget | 20 |
| Hindsight-feasible under arbitrary shape | 556 |

Of the incumbent's 151 failures, 101 are hindsight-feasible, 20
are already over budget before any observed harm, and 30 are
undefined. Therefore neither better optimization alone nor this
oracle can make the fixed roster pass all 606 views. There is still
room on supported recordings, but a deployable method must choose
where and how far to intervene. The next test should use a causal
recipient/amplitude rule against the same-budget simpler controls,
keeping the original readout and reporting undefined views.

## Reproduce

~~~sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_flight_shape_risk.py register
.venv-pytorch/bin/python scripts/run_m3w_opendd_flight_shape_risk.py run
.venv-pytorch/bin/python scripts/run_m3w_opendd_flight_shape_risk.py replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_fixed_radius_feasibility.py
~~~

50,400 updates and readout in 34.8 seconds on native arm64 CPU,
four Torch threads and zero workers. Checkpoints save optimizer
and draw state every 100 updates. Zero-update replay matches the
entire readout. Twenty-eight targeted tests passed. No CREATE job,
DroneCrowd confirmation, new deployment, Stage5C, or SMC.
