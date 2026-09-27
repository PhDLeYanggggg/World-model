# Crossed Fitting-Regime Study: Conclusions

## Verdict

The fixed experiment completed, but none of its mechanism screens passed.
Training-site regime and easy-definition changes do not provide a uniform
repair of cost transport. No new forecasting policy or model is promoted.
This is exposed source-development evidence, not independent confirmation
or a CCF-A submission-readiness result.

## What Was Actually Run

Fresh native Torch training:864 crossed cost heads,1,728,000 updates, plus
one2000-update native reproduction bridge. The bridge matches all original
parameters exactly. Cached controls:432 two-site and144 three-site heads,
with frozen parent magnitude readouts. All432 prediction replicas were fixed
and committed in23b75ca6 before held scoring. The144 held views completed
13,824 direct component-MSE checks. No unknown-label row was sampled.

Recorded fitting:1703.505 seconds including the bridge. Crossed training
heartbeats span43m35s after ancestry preflight. These are local CPU4,
interop1,workers0 runs, not NumPy fallback. Checkpoints and row-level scores
remain local. No remote job was submitted or modified.

## Main Evidence

All numbers below use the unchanged outer three-site easy definition.
Counts are positive/negative/overlapping-zero95% intervals across six
overlapping assignments. They are not six independent datasets.

| Full-input comparison | Interval counts | Assignment point range |
|---|---|---|
| Cut3 vs cut2, two fitting sites, raw | 1/1/4 | -5.29% to+1.82% |
| Cut3 vs cut2, three fitting sites, raw | 1/1/4 | -3.38% to+1.48% |
| Three vs two fitting sites, fixed cut2, raw | 4/0/2 | -18.10% to+6.72% |
| Three vs two fitting sites, fixed cut3, raw | 1/0/5 | -19.58% to+4.97% |
| Scaled vs raw, native two-site cell | 3/0/3 | +1.41% to+26.54% |

The first two rows reject a simple universal cut-definition repair. The
fitting-regime effect is conditional:4/6 favorable intervals atcut2 but only
1/6 atcut3, with a negative coverage-log guard interval in the latter.
The raw cut-by-regime interaction has1 positive and5 overlapping intervals;
the scaled interaction has6 overlapping intervals. A failed screen does not
prove an effect is exactly zero or identify a unique failure cause.

Size-matched scaling does not pass either. It improves some squared costs,
but its full-input coverage-log guard has6 negative intervals and top10 harm
capture has1 negative interval. Matching the number of fitting sites is
therefore insufficient to repair the measured magnitude/coverage tradeoff.
The regime still includes composition, preprocessing and sampling; this is
not a pure sample-size effect.

## Narrow Positive and Negative Findings

The larger regime at fixedcut2 has4 favorable full-input primary intervals.
After frozen scaling, larger-regime point estimates are all positive in both
cuts, but only1/6 and3/6 intervals exclude zero. Those conditional findings
remain visible; they do not meet the registered uniform-evidence screen.

Motion-only scaling improves primary MSE in5/6 native two-site intervals,
but its coverage guard worsens in3/6. Large negative raw motion-only relative
effects also remain reported, including a -824.11% assignment point in the
fixedcut2 regime contrast. No negative assignment is dropped or pooled away.
Full and motion-only have different forecast/event populations and are not
a matched feature ablation.

The separate mass diagnostic is post-hoc, descriptive, and not a gate.
In full-input native two-site records, median predicted/observed harm mass
falls from0.631 to0.097 after scaling. In native three-site records it falls
from0.663 to0.109. Duplicate native three-site replicas are removed. These
dependent summaries describe underestimation after shrinkage; they do not
establish a general causal explanation for the auxiliary failure.

## Decision and Limits

Keep deployment unchanged. Do not retune a cut, slope, threshold or checkpoint
against these results. Do not substitute the more favorable secondary-cut
diagnostic for the primary endpoint. Cost MSE is not trajectory ADE/FDE or
easy degradation; the coverage quantity is harm mass, not a conformal safety
guarantee. Three seeds and3000 four-locality resamples give descriptive,
unadjusted source-development intervals.

The next narrow question is why fitting-only squared-error scaling is so
poorly aligned with harm-mass estimation, even at a matched fitting regime.
That must be diagnosed before another training or calibration sweep.
Independent selection, reserved calibration and confirmation remain closed.
Obs8/pred12 native steps, detector pixels; no metric/seconds,human-gold,true3D,
foundation or physical-safety claim. Stage5C and SMC stay disabled.
