# Direct Forecasts With Past Traffic Context

**A new forecast improves on CV at fixed blend strength, but does not
qualify for deployment.** The most useful candidate here uses only
the pedestrian's own history. Extra observed motor traffic was not
helpful for this architecture. This is OpenDD development evidence.

## Matched Experiment

I changed the forecast rather than its ranking. A small network predicts
three bounded velocity knots in the pedestrian's local frame and integrates
them from the current point. The self-only, pedestrian-neighbor and
pedestrian-plus-traffic arms have identical model size, draws, loss and
training budgets. All 63 fits use three seeds and leave the evaluated
roundabout out of training. They share the existing 10,475 targets in
3,395 past-only queries, at the OpenDD-specific observation/forecast grid.

Past road input is newly extracted from hash-verified v3 SQLite archives.
Only requested past timestamps and position columns are read. Nearby motor
context is present for 5,132 targets (49.0%); no unavailable past samples
are filled from the future. The [publisher](https://l3pilot.eu/data/opendd.html)
documents projected coordinate units; this does not calibrate SDD or
DroneCrowd. It does not establish how the original tracks were produced.

## What Improved

| Forecast | Conservative gain vs CV, pp [95% site bootstrap] | Observed ADE / FDE | Worst descriptive easy ADE increase |
|---|---:|---:|---:|
| CV | 0 | 0.6083 / 1.3534 | 0% |
| Self-only, direct | +2.250 [-0.848, +6.653] | 0.5892 / 1.3002 | +56.29% |
| Self-only, fixed 25% blend with CV | **+1.792 [+1.029, +2.873]** | **0.5961 / 1.3264** | **+5.24%** |
| Pedestrian context, fixed 25% blend | +1.194 [+0.340, +2.190] | 0.5986 / 1.3303 | +11.49% |
| Traffic context, fixed 25% blend | +0.198 [-1.245, +1.579] | 0.6036 / 1.3387 | +23.92% |

Conservative gain debits every missing future step by forecast disagreement
and is normalized by observed CV error mass. The seven sites are weighted
equally after averaging seeds; intervals are from 3,000 paired site resamples.
ADE/FDE use observed positions in publisher units. The blend was fixed
before scoring. The self-only blend is positive at all seven held locations
after averaging seeds. Removing the smallest site descriptively leaves
+1.329 pp. The worst signed easy change occurs at rdb5 (+5.24%, 46 easy
targets); this is not an empty denominator. Positive easy harm is reported
separately from signed change in the machine-readable results.

## What Did Not Transfer

Adding pedestrian histories to the direct model costs -5.429 pp
[-8.508, -3.036] vs self-only. Adding motor context to the pedestrian
model costs a further -8.308 pp [-14.274, -2.743], and -14.152 pp
[-23.799, -5.488] within the past-defined motor-supported rows. The result
rejects this implementation of context, not interaction information in
general. The support stratum does not select on future motion.

In a separate fixed-count hindsight comparison, adding the self-only
forecast expands the original motion menu's conservative one-edit gain
by +1.451 pp [+0.911, +2.055], holding the exact same query roster. That
shows the new path is not merely another ranking of the old menu; it does
not show that a causal controller can find the right edits.

**Decision:** keep the current deployment. Carry the self-only forecast
forward to a separately specified train/inner-location calibration with
CV fallback. Retain the original easy-risk rule and compare at equal edit
counts against the old bank. OpenDD is already exposed to development;
positive evidence here must not open the 30 sealed DroneCrowd groups.
No Stage5C, SMC, multimodal generalization or submission-readiness claim.

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_location.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_velocity_candidate.py
```

The fresh run took 33.48 seconds including traffic extraction on arm64 CPU,
four compute threads and no workers. Checkpoints are written every 100
updates; a site-bound cache and PID heartbeat allow recovery. The verified
checkpoint readout matched byte for byte. No source rows, caches or model
weights are distributed.

[Protocol](protocol.md) | [Results and loss traces](analysis.json) |
[Anonymous supplement](../evidence_manuscript_v28/README.md)
