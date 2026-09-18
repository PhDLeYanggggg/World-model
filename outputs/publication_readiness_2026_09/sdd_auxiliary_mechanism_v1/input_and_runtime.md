# Source Mechanism Controls: Verified Inputs and Live Training

Result source: fresh source-support/permutation diagnostics and real Torch pilot;
cached_verified source cache and 54 parent control fits. Full new matrix still
running at this checkpoint; this page is not a completed model result.

Registration SHA256:
`699c9f6fdb346b828dee4411894b2747a05f541e404285325fa0a93f2174837a`.
Design/code frozen in `bba7fbe1` before real fitting. Twenty focused tests pass,
including exact main-only resume and same main-stream checks. Full legacy suite
not rerun. Runtime is native arm64 CPU, Torch 2.12.0, NumPy 2.4.6, four compute
threads, one interop thread and no DataLoader workers.

The source-label-permuted RGB pilot completed 100 actual updates in 4.444 seconds
and saved a checkpoint without evaluating held fit rows. Pilot PID 80309 exited 0.
Full trainer PID 80360 / tool session 81910 resumes that same fit, not a fresh
replacement. Private heartbeat and checkpoints are under
`data/stage_cvpr2027_experiments/sdd_auxiliary_mechanism_v1/`.
The fixed budget remains 54 new fits / 270,000 updates, with 54 old controls.

## Permutation Evidence

All 229,333 source rows are mapped bijectively within original recording and
exact future-support pattern. Original train-40 membership is unchanged.
All three seed maps preserve the support mask. There are 197 singleton rows,
and 5,379-5,442 donor pairs retain the same recording-local track. The ablation
does not erase all dependence. Independent real-input checks verify 128 rows
per seed: inputs and predictions remain exact across geometry/mask/RGB even
when source loss labels are permuted. The donor map never enters inference.

## Source Support and Target Scale

On 188,358 complete-label source windows, the broad exact-static-to-any-motion
category has 10,039 windows / 342 recording-local tracks. The prior audit's
stricter half-past-box-displacement proxy has only 244 train-only windows / 58
tracks, matching the same-source historical census. These are different labels;
the earlier all-60 count must not be compared as if it used the same population.
The 40,975 partial/absent-label windows remain in training and are not assigned
a complete-label event category here. No sampling or training label changes.

The fixed native-coordinate scale floor is reached by 22,374 complete SDD windows.
In the broad static-moves category, normalized mean displacement has median
1,125 in SDD versus 24.702 in the fold-0 main training cohort (Hotel/grouped Zara).
This is not a physical-unit comparison or proof of a behavioral difference.
At the CV prediction, the median scalar log1p-loss sensitivity dL/dADE is
0.000888 for source static-moves, versus 0.038907 for the corresponding main
training category and 0.868029 for source moving categories. These are analytic
scalar loss derivatives, not measured neural parameter-gradient norms.

Thus a fixed 0.001 native-unit floor does not make stationary targets invariant
to changing pixels into another coordinate unit. This is a concrete diagnostic
and a possible mechanism for weak source learning, not a proven explanation of
all held-scene failure. The running experiment is unchanged; no primary-metric
repair, loss-weight change, threshold search or closed-label access is made.

See [source support](source_support.json) and [input checks](input_checks.json).
No independent confirmation, deployment, metric/seconds, true-3D or foundation
claim. Stage5C and SMC remain disabled.
