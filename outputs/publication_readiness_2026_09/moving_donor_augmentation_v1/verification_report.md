# Execution Verification And Limits

Definition commit:32a2010f, before real donor preparation or primary fitting.
Nativearm64 .venv-pytorch, CPU4/inter-op1/workers0; no new CREATE job.

| Phase | PID | Observed exit | Seconds | Peak RSS bytes |
|---|---:|---:|---:|---:|
| prepare | 63754 | 0 | 158.920 | 2591277056 |
| real recovery pilot | 63978 | 0 | 5.115 | 4262133760 |
| 24 primary fits | 64054 | 0 | 153.805 | 5659852800 |
| 24 full refits | 64455 | 0 | 152.364 | 5655592960 |
| final development readout | 64729 | 0 | 8.255 | 3740434432 |

Every primary model completes4,096updates; all24repeat exactly, including
model/optimizer/RNG state, trace and sampling visits. Recovery128+128vs256also
matches exactly. The primary total is98,304updates, full repeats another98,304,
and the pilot512. This is not32epochs or three-seed scientific confirmation.
Direct checkpoint inspection also confirms first and last layer weights change
in24/24models, every optimizer parameter reaches step4096, and every model draws
exactly2,097,152training samples with replacement. Query sources are absent from
all recorded fitting-source lists. These draws are not independent data records.

36focused tests pass in3.27s. The full historical suite was not rerun. Source
exclusion, held-label preprocessing invariance, input/label separation, exact
resume and no-deployment-overclaim checks are included. Tests do not by themselves
prove scientific improvement; the separate final own-B readout supplies that
limited component evidence.

## Initial Replay Failure

PID64780 completed all numeric prediction/metric comparisons exactly, then
exited1 because the reporter concurrently added public report files during the
public-directory invariance snapshot. The full traceback and events are preserved
in the local execution history. No optimizer update or training restart was used
to address this reporting race. The final replay is run serially only after all
reports stabilize; its observed receipt is stored in the private experiment
directory. Future reproduction must not run reporter and replay simultaneously.

Serial replay PID65165 subsequently exits0 in8.120s, peak3,747,692,544bytes.
All24prediction arrays and complete metrics match; all public files present
during that snapshot remain unchanged. Fitting is explicitly forbidden. This
is verified replay, not independent confirmation. The final figure is visually
checked locally. No training, verification, evaluation or replay PID remains live.
The compiled main-paper SHA remains unchanged:
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.

## Scientific Status

The matched donor-information screen passes3/3, not the original full-policy
safety gate. Relative to the forest, false-safe cost increases in3/3contexts.
Source060's harmful-admission count worsens despite better total decision cost.
No outer C/H evaluation or model promotion was performed. One seed, exposed
development records, candidate/teacher-size confounds and no new confidence
intervals mean this cannot stand alone as a submission-ready main result.

Original2%/support requirements, empty-control NA, old failures and24-group main
NA remain. Source annotation-generation causality is unverified. No verified
metric/elapsed-seconds, true3D, foundation, Stage5C or SMC claim.

Analysis SHA256:
`394c99ce6879f894fb43c10fb7caee58782c283b2e897771dec3dff7bb2e1ee9`.

Code/config/reports/light metrics only are uploaded. Raw labels, donor arrays,
forecast caches, checkpoints and figures stay local. The unrelated staged-index
digest remains`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
