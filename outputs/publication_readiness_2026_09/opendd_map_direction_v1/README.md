# Static Road Context for Matched Path Selection

The [OpenDD release](https://l3pilot.eu/data/opendd.html) includes a
drivable-area visualization and an affine world file for each roundabout.
I joined these static assets to the past coordinates in the existing
seven-location cohort. No tracks were used to build the maps. These
maps describe vehicle space, not sidewalk connectivity, legal crossings,
pedestrian intent, collision risk or physical safety.

All map members and world files are hash-bound. The PNGs are 3840 x
2160 grayscale, with 0.10-0.17% intermediate edge pixels; a fixed
midpoint binarizes the small fringe. Both affine axes are applied in
world-file order; distances use axis scales, bilinear raster samples and
a near-orthogonal approximation. No out-of-raster points are padded.
Of 30,075 edited row-seed cases, 26,472 (88.0%) support all past and
predicted samples; 4,821 show an observed sign crossing. Unsupported
cases retain matched hindcast. No TUMDOT map transfer was run.

At inference I select among the same six paths at identical recipients
and per-agent peak displacements. Persistence favors maintaining the
last signed distance to the road. Continuation extrapolates the four
latest observed signed distances with a slope capped by past speed.
The 8-to-12 protocol and original per-recording 2% rule are unchanged.
There is no new training, target tuning or independent confirmation.

| Half radius | Lower vs CV (pp) | Direct vs hindcast (pp) | Direct vs fixed tangent (pp) | 2% passes |
|---|---:|---:|---:|---:|
| Matched hindcast | +0.0056 | 0 | -0.3645 | 552/606 |
| Fixed tangent | +0.3057 | +0.2494 | 0 | 522/606 |
| Map persistence | +0.2047 | +0.1904 | -0.1565 | 541/606 |
| Map continuation | +0.1857 | +0.1730 | -0.1776 | 536/606 |

Continuation's direct loss versus tangent is [-0.2236, -0.1299] pp.
The 30 undefined easy views remain undefined; 40 other views fail for
continuation. Quarter radius still loses to tangent (-0.0899 pp) and
passes 561/606, below matched hindcast's 565/606. The intervals use
3,000 resamples of seven locations, after averaging three seeds.

## Isolating assignment rather than action mix

An exact permutation expectation holds each action count, recipient
and peak radius fixed. The original control conditions on recording and
seed. Two follow-up controls also preserve map support, and then support,
last observed road side and observed crossing status. Strata are all
past- or forecast-defined; this is utility attribution, not a risk
certificate for shuffled policies.

| Continuation vs same-count shuffle | Direct observed (pp) | Direct conservative (pp) |
|---|---:|---:|
| Within recording and seed | +0.0324 [0.0109, 0.0550] | +0.0033 [-0.0230, 0.0340] |
| Also preserve map support | +0.0273 [0.0064, 0.0480] | +0.0187 [-0.0024, 0.0386] |
| Also preserve road side and recent crossing | +0.0396 [0.0148, 0.0657] | +0.0308 [0.0051, 0.0592] |

The third comparison gives a small **secondary development signal**
beyond coarse map state, not a pass on the prespecified overall
comparison. Persistence has no clear conservative assignment gain.
Map continuation still loses to fixed tangent and fails the original
risk gate. No deployment change, risk redefinition or opening of the
DroneCrowd confirmation groups. This motivates testing pedestrian-
specific route geometry or a training-side map feature with a
matched ablation, not using this heuristic as a deployment policy.

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_map_direction.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_map_direction.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_map_support.py
```

[Protocol and map hashes](registration.json) | [Main diagnostic](analysis.json) |
[Map-conditioned comparisons](support.json).
