# Complete-Region And Minibatch Gradients Differ

## Actual Frozen-State Check

Fresh TRAIN-only diagnosis,54current frozen heads,zero optimizer updates.
PID97446exit0 takes84.567688s,peakRSS10,045,472,768B. Separate PID97916exit0
replay takes86.062640s;all54group statistics and the summary reproduce exactly.
Every original2,000-step sampler hash matches its training endpoint. No held
recording or independent role is opened. Seven initial focused tests pass1.32s;
the expanded exact-region/recovery suite passes10tests in1.70s.

## Measurement

Each frozen head compares64evenly indexed original minibatch gradients with
the complete fixed-region divergence gradient. The latter weights nonempty
teacher-selected known-label recording regions equally. The former averages
represented regions' nonlinear sample-mean loss. They are not algebraically
identical estimators of the same finite objective.

| Bank | States | Median cosine | Median relative vector difference | Steps without auxiliary region |
|---|---:|---:|---:|---:|
| Physical-only | 18 | 0.650740 | 0.971071 | 0/36,000 |
| Mixed | 18 | 0.685630 | 0.940755 | 1/36,000 |
| Neural-only | 18 | 0.662788 | 0.877989 | 5,500/36,000 |

Actual auxiliary-region visits over all2,000steps vary sharply. Physical-only
has3zero-visit regions among420repeated region/head views,median97.5visits;
mixed has7/425zero-visit regions,median81;neural-only has6/381,median53.
Some regions are represented on almost every step while others are never
represented. A zero auxiliary-region visit is not proof that the entire
recording was absent from ordinary pointwise training.

The mixed base-versus-exact-EH gradient cosine median is+0.117095,and the
other-regional-moments-versus-EH median is-0.003343. Strong systematic gradient
opposition is not established by these summary values. The64-batch mean
contains Monte Carlo noise,and region inclusion weighting and nonlinear
sampling are also different. This is measured finite estimator mismatch,not
a theorem proving infinite-sample bias or the sole cause of held failure.

## Chosen Repair

Test cyclic complete fixed-region moment training on the same network and
teacher centroids. Each update retains the original base minibatch,then adds
the exact gradient for one complete teacher-selected known-label recording
region. Cycle a fixed seeded region order. Stream two passes with chunk512
to bound memory,then clip/step once as before. Count additional teacher rows
and forward work separately. Matching base draws and steps does not match
FLOPs,and balancing regions changes their old implicit representation weights.

The full-vs-chunked gradient equality is tested against direct autograd on a
synthetic multi-region case with a zero target. Activation,real recovery and
disabled neutral must be tested before primary training. No forecast,input,
normalizer,teacher choice,coefficient or inference threshold is tuned from
held outcomes. This repair has not passed a utility or safety gate merely
because the diagnostic motivates it.

## Boundaries

The original scene/region heads and all negative results remain unchanged.
Old pointwise-deviance,posthoc recalibration and SDD adaptive reweighting are
not rebranded as new successes. This diagnosis concerns the current398-feature
cost head,not the older compact19-input decision-risk head.

Obs8/pred12/rawstride12,EuropeanSquares SOURCE webcam detector-track pixel
coordinates,not metric/seconds,gold,true3D or foundation evidence. The source
inner-held cohort is exposed development,not independent confirmation or
whole-pipeline OOF. Empty risk is not_estimable;count matching is utility-only.
No deployment,Stage5C execution orSMC.

```bash
.venv-pytorch/bin/python scripts/diagnose_m3w_region_moment_gradients.py verify
```

Private checkpoints/source arrays are required. The public54aggregate files
contain regional means and gradient statistics,not trajectory rows or weights.
