# Learned Map Context at Matched Intervention Radius

I trained three equal-width path-value heads: motion only, correct
road-map features, and wrong map features. The wrong-map control
exchanges descriptors only between mapped agents in the same query.
Both map arms use zeros when no exchange is possible. There are
24,801 such eligible target-seed cases out of 30,075 edits at half
radius. The map is the publisher's static vehicle-area raster, not
a pedestrian map or a physical-safety label.

The three heads see the same training draws, initialization,
candidate menu and loss. All producers, refiners, head training and
easy thresholds exclude the evaluated location. Donor candidates use
producers that also exclude their own donor location. The donor-side
refiner can have seen that donor; only outer-location exclusion is
claimed for the full pipeline. Static assets of the evaluated location
are available at inference, but its future labels are used only to
score. These are exposed development sites, not an independent test.

The loss predicts paired gain and positive easy harm per unit radius.
The prediction changes path only if utility and harm-penalized rankings
agree and the proposed choice beats fixed tangent on predicted penalized
utility. Otherwise it keeps tangent, still at the same radius. The rule
is not a calibrated safety certificate. There are 63 native Torch fits,
three seeds and 31,500 optimizer steps with four CPU threads and zero
workers. No new CREATE job was needed.

| Half radius | Lower vs CV (pp) | Direct vs fixed tangent (pp) | 2% passes |
|---|---:|---:|---:|
| Fixed tangent | +0.3057 | 0 | 522/606 |
| No map | +0.2907 | -0.0210 | 526/606 |
| Correct map | +0.2899 | -0.0242 | 527/606 |
| Wrong map | +0.2773 | -0.0357 | 529/606 |

Real map versus no map gives **-0.0059 pp** [-0.0466, +0.0287] directly;
real versus wrong map gives **+0.0081 pp** [-0.0178, +0.0360]. Correct
map versus its own same-count shuffle is +0.0295 pp [+0.0126, +0.0483],
holding recording, mapped-query support, recipients and radii fixed.
But the **paired difference of shuffle advantages** versus motion-only
is +0.0195 pp [-0.0075, +0.0447], and versus wrong-map is +0.0149 pp
[-0.0055, +0.0359]. A positive interval for one arm alone does not
establish an advantage over another arm.

With the **same map-trained weights**, exchanging the map at inference
changes 7,162/24,801 eligible decisions. Correct versus exchanged map
is +0.0144 pp [-0.0100, +0.0371] directly; setting the map channels to
zero changes 8,772 decisions, with +0.0149 pp [-0.0094, +0.0386] for
correct versus zero. Thus the head uses the map, but improved utility
from correct alignment is not established. All intervals use 3,000
paired location resamples after averaging the three seeds. At quarter
radius correct map remains below tangent in utility and passes 557/606
versus 560/606 for tangent. Thirty views remain undefined throughout.

**Decision:** retain fixed tangent as the stronger matched utility
reference. The results do not support a primary claim for this raster
representation, a risk pass or a deployment change. No independent
confirmation, Stage5C or SMC. Future map work needs pedestrian-specific
geometry or a prespecified causal representation that can be evaluated
against the same strong controls, rather than more tuning on this set.

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_map_path_value.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_map_path_value.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_map_value_counterfactual.py
```

[Training protocol](registration.json) | [Held-location readout](analysis.json) |
[Frozen-weight perturbations](counterfactual.json).
