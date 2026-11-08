# Learned Forecast Shape at Identical Recipients and Radii

**Positive shape-only development result, not a qualifying controller.**
The [geometry study](../radius_matched_turn_menu_v1/README.md) holds all
1,457 edits, recipient times and peak CV-relative radii fixed. Its six
causal paths differ in correction timing and direction. The analysis
uses the same seven exposed OpenDD locations and 202 distinct recordings,
eight observed and 12 forecast native steps, and unchanged 2% risk.

We train two heads with the same architecture on fitting locations
only and all causally eligible rows at three fixed bridge amplitudes.
The value head predicts each path's conservative gain over the linear
edit and original risk terms. The dynamics head predicts a
heading-aligned future displacement relative to CV, masking missing
labels in the loss. At inference it chooses the closest candidate in
mean trajectory distance; it never emits the unconstrained internal
prediction. Neither head can change who is edited, when, or the radius.
An earlier CV-tangent forecast is a further no-refit control: its
direction is rescaled to exactly this roster's radii, not scored at
its original admission rule or edit coverage.

| Shape rule | Original risk passes | Risk pass and beat both |
| --- | ---: | ---: |
| Frozen linear | 87/606 | 61/606 |
| Frozen neural tangent, rescaled | 96/606 | 71/606 |
| Expanded bank, observed-prefix choice | 112/606 | 77/606 |
| Learned gain only | 124/606 | 89/606 |
| Learned gain with risk penalties | 114/606 | 77/606 |
| Learned dynamics proxy | 109/606 | 83/606 |

For the dynamics proxy, direct full-grid lower gain is **+0.032 pp
[+0.009, +0.055]** versus linear, and **+0.078 pp [+0.007, +0.177]**
versus the rescaled tangent. Against the expanded prefix selector it is
**+0.004 pp [-0.060, +0.059]**. The gain-only head has more dual-positive
views but does not improve direct gain over the anchor (-0.008 pp
[-0.058, +0.035]). These distinctions matter: neither a larger pass
count nor a positive contrast to the linear anchor proves superiority
to the best simple rule. Intervals resample locations after averaging
seeds, not windows. All contrasts and units are development-only.

All arms have **334/606** undefined easy-risk views. The joint gate
does not pass. The earlier tangent control has no zero directions on
the selected roster. Next test whether past-only admission can improve
easy support while preserving forecast utility; do not promote the
dynamics proxy as a deployed controller. The 30 DroneCrowd confirmation
groups remain sealed; Stage5C and SMC remain off.

`fresh_run`: 42 arm64 CPU Torch fits, 33,600 updates, three seeds,
four threads, no workers; and the equal-radius tangent audit.
`cached_verified`: original controllers, checkpoints and zero-update
replay. [Metrics](analysis.json) | [Tangent control](tangent_control.json).

```sh
.venv-pytorch/bin/python scripts/run_m3w_shape_menu_value.py
.venv-pytorch/bin/python scripts/run_m3w_shape_menu_value.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_shape_tangent_control.py
```
