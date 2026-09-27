# Fixed-Floor Risk Heads: Model and Data Card

## Intended Use

Development-only cost/risk estimation for selective neural trajectory use.
Not a new trajectory forecaster, autonomous controller, safety certificate or
deployable model. The protected causal floor remains the deployment reference.

## Material Passport and Population

Fresh216 neural risk heads; cached_verified nine unit-repaired forecaster
banks and complete floor/utility producer chains. Twelve opened European source
localities. Detector-derived silver trajectories in image-local coordinates;
obs8/pred12 rawstride12. Source roles are4 forecaster-fit,4 floor-controller-fit,
2 risk-fit,2 risk-readout, disjoint per context. Three forecaster seeds17/29/43;
all six risk-fit/readout pair rotations. Views share sources and windows.

No final independent selection, calibration or confirmation population was
accessed. Repeated use of development localities prevents treating these
readouts as independent external tests.

## Inputs and Outputs

380 causal features; frozen fit-only source-balanced means/std, cost scale and
clip10. Inputs include past motion/interaction and current candidate-rollout
diagnostics, not future target endpoints, validity masks, target latents,
central velocity or test-derived goals. Future labels are loss/evaluation-only.

Architecture380->64 GELU->4, 24,644 parameters. Outputs are all/easy reference
cost via softplus and all/easy harm via sigmoid times a causal maximum-rollout
disagreement envelope. The envelope is a geometric bound on candidate-error
difference, not a bound on real physical harm or a confidence guarantee.

Ordinary MSE and fitting-positive90th-percentile tail4 weighting share all
other training factors. Tail weighting changes the conditional estimand; it is
not an automatically calibrated expected-harm estimate. Unknown-label rows
never enter supervised sampling; they remain in inference and count matching.

## Optimization and Recovery

Fixed2,000 updates, batch256, AdamW lr0.0003/weight_decay0.0001, gradient clip5.
Native arm64 CPU4, interop1, no DataLoader multiprocessing. Atomic500-update
checkpoints include model, optimizer, sampling and Torch RNG, draw counts,
training monitor and input provenance. No held early stopping or checkpoint
selection. The first100-update pilot is inside the prescribed budget.

Frozen positive utility, movement and99% fitting-support guards precede fixed
2% all/easy predicted-risk screens. Rejection returns to protected damping.
Same-frame matched-count MSE selection is a diagnostic control, not a rule
certified to meet its own predicted budget.

## Evaluation and Limitations

Twelve source-mean bootstrap units;3,000 draws. Dependent views are averaged
before source resampling. Unadjusted development intervals are not
simultaneous or independent-confirmation intervals. Three seeds belong to the
frozen forecasters and their deterministically matched head fits, not three
new whole-world-model trainings.

All registered controls, empty selections, easy degradation, selected harm,
partial labels and unknown-label interventions are reported. No cold raw-data
rebuild or full legacy test suite is claimed. Private weights/caches/raw data
are not uploaded. No metric/seconds/true3D/foundation/physical-safety claims;
Stage5C and SMC remain disabled.
