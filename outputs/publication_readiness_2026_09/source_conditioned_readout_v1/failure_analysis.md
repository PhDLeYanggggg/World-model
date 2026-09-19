# Failure Analysis: Better Conditioning, No Forecast Advantage

## What the Intervention Established

The 24 fixed fits complete, with no change to output support, prediction class,
loss, data, draw streams or evaluation. The pre-bound output is scaled by a
training-complement constant. Logged gradient clipping falls from 100% to 0%
for all models. Geometry/centered equal-site gains improve relative to matched
controls by +0.031806/+0.273252 percentage points; both conditional contrast
intervals are positive. This is a measured optimization/control improvement.

## What Still Fails

Both arms remain below stationary CV: -0.000251% and -0.001342%, respectively.
All 24 held fits are negative. Their nonzero-target gains are also negative,
about -0.000055%/-0.000065%. Small positive hard-slice numbers are many orders
below the requested practical improvement and coexist with negative whole-site
results. Centered appearance remains worse than geometry.

Static harm shrinks to 0.00000644/0.00004117 annotation pixels. These tiny
differences must not be interpreted as physically meaningful precision or
world dynamics. Static CV already has zero error; percentage easy degradation
is undefined. A tiny number is not evidence of passing the 2% rule.

The fixed forecast/CV binary-oracle gains are only 0.0000303% and 0.0003394%.
These ceilings concern these specific candidate forecasts, not every possible
future model. Another threshold sweep over the same candidates cannot provide
substantial recoverable gain.

## Failure Taxonomy

| Hypothesis | Evidence | Conclusion |
| --- | --- | --- |
| Mean gradient direction is reversed by clipping | Training-only frozen audit gives clipped/population cosines >=0.99858 and >=0.99930 | Large final-iterate reversal unsupported; earlier/Adam effects not ruled out |
| Output parameterization magnifies small steps | Train-derived gains 527-627 remove logged clipping and most tiny jitter | Optimization behavior changes, not a prediction success |
| Suppressing static error alone restores dynamics | Static error approaches zero but nonzero-target gain stays negative | Repair is primarily closer approximation to the baseline |
| More motion in the marginal prediction helps | Fully static training futures exceed 50%; constant-movement triangle bound is positive | Conditional past evidence is required; this is not a proof of unlearnability |
| Frozen centered image features supply usable extra signal | All sites remain negative; centered-minus-geometry contrast is negative | No demonstrated benefit of this particular visual representation |
| Lack of independent transition support limits generalization | Prior verified audit has only 47 tracks with half-box excursions in this subset | Plausible limitation, not established sole cause |

The Euclidean loss is nonsmooth at exact static predictions. A large final
gradient near zero does not by itself prove a faulty optimizer or implementation.
The registered repair isolates output parameterization, not a pure clipping
change; effective learning rates and weight-decay geometry can also differ.

## Next Discriminating Work

Do not spend another fixed matrix on clip thresholds or rescaling these same
near-zero forecasts. Inspect whether raw past visual motion retains information
lost by frozen image pooling, with strict past-frame alignment and the same
approved source role. Distinguish visible motion from camera/crop displacement,
occlusion and annotation interpolation before a new representation comparison.
Register any subsequent fit before scoring; keep all source development results
exploratory and leave main/outer/independent confirmation untouched.

This route is not yet run and is not promised to succeed. The hypothesis that
more informative past observations improve conditional prediction remains open.
No new deployable neural model, Stage5C or SMC is authorized by this result.
