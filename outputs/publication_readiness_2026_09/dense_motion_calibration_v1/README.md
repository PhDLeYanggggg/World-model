# Broad Motion Improves Coverage; Learned Risk Does Not Transfer

Three fresh Torch models fit on the original 69,746 rows from nine
source groups, 1,000 updates each. No changes to the CV/OLS4 paths.
Every decision is one of five fixed interpolation amplitudes. The
model sees eight positions and internal hindcasts only, and learns
gain, positive harm, and all/easy reference mass from the fitting
labels. On the 40 already opened DroneCrowd calibration recordings,
all new choices freeze before the new outcome readout. This is not
new independent calibration or a deployment decision.

| Policy | Changes / requested seed-rows | Lower gain vs CV, pp [95% group interval] | Actual 2% passes / 120 |
|---|---:|---:|---:|
| Full OLS4 | 75.50% | +51.288 [+46.487, +55.878] | 24 |
| Fixed half blend | 75.50% | +32.643 [+30.303, +34.838] | 102 |
| Independent learned gain | 75.23% | +47.228 [+43.600, +50.609] | 69 |
| Recording price ladder | 0% | 0.000 [0.000, 0.000] | 0 |
| Predicted-feasible union repair | 4.34% | +0.813 [+0.538, +1.131] | 99 |
| Matched past-hindcast allocation | 4.34% | +0.898 [+0.599, +1.234] | 114 |

The independent arm changes 3,966,650 forecast paths but its direct
conservative contrast to OLS4 is -4.120 pp [-5.439, -2.823].
The original reference risk definition is unchanged; CV identity
has undefined selected denominators and is not a pass. Equal counts
do not imply equal path displacements. Gain is normalized by the
observed CV error mass; intervals resample 15 source groups after
averaging the seeds.

## Fixing the Search Does Not Fix the Risk Estimates

A new input-only audit found positive-gain singleton actions that
pass both predicted ratios in **all 120** views. The price ladder's
empty output was therefore not evidence that the predicted set
was empty. The follow-up uses the same heads, picks the largest
positive predicted gain among individually feasible actions on
each row, and joins them. Summing each linear numerator/denominator
inequality preserves predicted feasibility. This is a constructive
repair, not an optimizer guarantee or a probability certificate.

The repair yields nonempty predictions in all 120 views and
228,655 changes, before labels are read. Predicted worst all/easy
ratios are 1.08% / 1.84%; actual worst ratios are 9.50% / 7.55%.
All 21 failed views exceed the actual all-risk threshold, and
three also exceed the easy positive-harm ratio. The matched
hindcast control keeps each query's exact multiset of amplitudes,
allocating them by a short past-only CV/OLS4 comparison. The learned
union's direct contrast to it is -0.08865 pp [-0.13278, -0.04745].
It loses 17 original risk passes and gains two, for a net loss of 15.
The matched control is conditioned on the learned amplitude allocation;
it is not a standalone independently calibrated policy.

**Decision:** Do not deploy. Broader motion coverage is viable on this
roster, but the learned dense moment estimator does not improve the
matched allocation and underestimates risk on some recordings. Fixed
half remains a necessary broad-coverage control, not a passing policy.
The next technical comparison should test the simple causal control
before adding a neural adjustment. Do not tune a new margin on these
outcomes or lower the gate. Confirmation remains sealed.

This is a specific transfer diagnostic, not a claim to implement
[multi-expert deferral's consistency guarantees](https://research.google/pubs/regression-with-multi-expert-deferral/)
or the [Smart Predict-then-Optimize loss](https://arxiv.org/abs/1710.08005).

## Replay

~~~sh
p=scripts/run_m3w_dense_motion_calibration.py
for phase in register train infer score replay; do .venv-pytorch/bin/python "$p" "$phase"; done
.venv-pytorch/bin/python scripts/audit_m3w_dense_predicted_feasibility.py
p=scripts/run_m3w_dense_safe_union.py
for phase in register infer score replay; do .venv-pytorch/bin/python "$p" "$phase"; done
.venv-pytorch/bin/python scripts/report_m3w_dense_motion_transfer.py
~~~

Native arm64 CPU, four compute threads and zero loader workers. Training
checkpoints every 100 steps, and the inference and score phases resume
complete recordings. Both no-label scalar replays matched. Twelve
focused tests passed. Private checkpoints and paths stay local.
[Protocol](protocol.md) | [Initial results](analysis.json) |
[Predicted feasibility](predicted_feasibility.json) |
[Repaired readout](../dense_safe_union_v1/analysis.json) |
[Verification](../dense_safe_union_v1/verification.json).
