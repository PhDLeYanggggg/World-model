# Execution And Evidence Verification

## Observed Execution

- Protocol/code registration committed63533cc8 before data packing and training.
- Recovery-verifier correction committed0d5dc833; original pilot failure retained.
- prepare PID51230: exit0, 125,540 rows, 8.906s.
- original pilot PID51282: exit1 after the direct and resumed fits completed.
  Difference: logs[1,20,40] versus[1,40], not model/optimizer/RNG/draw state.
- saved-state verifier PID51408: exit0, 0.888s; all training state, common loss
  records and prediction outputs exact. No additional optimizer updates.
- train PID51455: exit0, 8133.811s, peak1,868,922,880bytes.
- forecast PID61091: exit0, 549.829s, peak1,197,572,096bytes.
- replay PID61818: exit0, 573.711s, peak1,134,657,536bytes.
- Reporter exits0 and the local two-panel training/donor figure is visually checked.
- 24 focused tests pass in3.20s. Full historical test suite not rerun.
- No live train/forecast/replay PID remained at completion. No new CREATE job.

All six models complete4,000updates. The primary24,000updates include the
reused40-update pilot prefix; a further40updates belong to the direct recovery
control. This is one seed, not three seeds or independent confirmation.
All125,540global row IDs occur exactly once in the OOF bank. Nonfitting-source
draws are zero. All six saved forecast arrays replay exactly. Unknown futures
remain masked:2,824wholly unknown,122,716some known,90,965complete.

## Boundaries

Fresh fitting and donor generation do not establish a new safe policy. No risk
head was fitted on these new donors yet. No outer C/H or protected test was
newly evaluated. Per-parent training exclusions are retained even though a
source can have a different role in another parent context.

The source annotation-generation process is not proved temporally causal;
the model interface is causal by annotation index. All units remain image
coordinates/native annotation steps with unverified elapsed seconds.
Original2% risk/support rules, signed easy-net requirements, empty-control NA,
old failures and original24-group main NA remain. No identity certificate,
deployment change, true3D, foundation, Stage5C, SMC or submission-ready claim.

Analysis SHA256:
`74f9a3e0bf40e11dec548b87fa2d1450dbe6548e051a42b97bc4880a12e8cd21`.

Unchanged compiled main-paper SHA256:
`d9babe3ddbac858bcc1ac30e3ba98b332bed625c78ff9231d1d42a3e30b11772`.

The unrelated staged-index digest remains
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
Only scoped code/config/reports/light metadata are committed. Original labels,
arrays, forecasts, checkpoints and figures remain local. A replay is not a new
independent scientific result.
