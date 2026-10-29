# Static Context Audit: Usable Mapping, Not a Proven Visibility Mechanism

Fresh TRAIN-only calculation and separate exact replay,zero optimizer updates.
The original launcher failed with `KeyError: source_contexts` before any record
view or result;`attempt_v1.json` retains that failure. Corrected execution uses
the parent's actual contexts object,registered in `registration_v2.json` before
running. Scientific scope and features are unchanged.

The real run completes148context/record views over74unique fitting recordings,
14actual camera polygons and149,510unique past-eligible rows. Repeated context
rows total299,020,not299,020independent observations. Runtime18.434090s;exact
replay18.855576s. Each raw camera is bound by file ID. Locality007includes
cameras7and45,andlocality067cameras67and68;using one polygon per locality
would be wrong. No current point lies outside the broad normalized polygon
bbox[-.5,1.5]. This is a gross compatibility check,not calibrated alignment.

The original motion packing is exactly invariant to global image translation
in the synthetic regression check. The new18-field static/past/rollout function
distinguishes different positions within a fixed region and rejects a future
endpoint argument. It does not consume future support,targets,scene IDs or
test statistics.13focused scene/base/readout tests pass2.24s,including the new
scene-head exact-recovery controls;the full historical suite is not rerun.

## Fixed TRAIN Diagnostic Bins

Distance is current box-midpoint signed distance to the publisher's place
polygon,normalized by its bbox diagonal. Bins are fixed prospectively. Counts
repeat contexts;these are descriptive fitting outcomes,not a generalization test.

| Distance interval | Row views | Unknown outcomes | Unknown rate | Mean known floor ADE,pixels |
|---|---:|---:|---:|---:|
| below-0.10 | 40,964 | 820 | 2.002% | 13.560 |
| [-0.10,0) | 90,032 | 1,846 | 2.050% | 18.276 |
| [0,0.02) | 27,498 | 786 | 2.858% | 17.265 |
| [0.02,0.05) | 37,304 | 870 | 2.332% | 22.577 |
| [0.05,infinity) | 103,222 | 2,004 | 1.941% | 20.310 |

There is **no monotonic missingness gradient**. These data do not establish
that region edges cause the policy failure. Camera,record and motion confounding
remain. A polygon may cover only a place of interest,not the full image;the
publisher also uses a different processed footpoint convention. We retain raw
box midpoint targets and do not clamp trajectories or filter outside rows.

The justified next test is a registered feature ablation with exact disabled
control,not a learned visibility guarantee or new threshold picked from bins.
Independent roles and deployment remain unchanged. Raw pixel/frames only;
no metric/effective seconds,walkability,true3D,foundation,Stage5C orSMC claim.

```bash
.venv-pytorch/bin/python scripts/audit_m3w_static_scene_context.py verify
```

This command recomputes the diagnostic using cached-hash-verified existing
source assets. It does not retrain a forecaster or independently confirm a score.
