# Stationary-Start Label Resolution: What Was Ruled Out

## Scope and Provenance

Two fresh fit-only diagnostics use the unchanged 365 stationary windows, with
31 agents and 45 runs at ETH and Hotel. They are not independent test windows.
The original eight-to-twelve-step task, primary metric, labels and source roles
are unchanged. No model was newly trained, selected or deployed in these audits.
The 72 corrected classifier/regressor artifacts were hash-verified and replayed;
the largest saved-prediction difference was 2.22e-16. No development, calibration
or confirmation labels were opened.

The [resolution audit](results.md) binds its source, geometry and previous model
reports. The [conditional CV audit](../quantized_cv_feasibility/results.md) binds
that result and its own solver implementation before execution. Per-row data and
model files remain local. Public JSON contains aggregate counts and metrics only.

## Findings

1. All cached native positions match the original text. None of the 188 changed
   future windows can be explained as a constant coordinate under the printed
   decimal intervals. Printed precision is not measurement uncertainty.
2. Under supplied H, all 8,908 ETH and 6,544 Hotel source positions are within
   0.001 inferred pixels of an integer lattice. This identifies numerical
   lineage, not independently verified image registration or physical scale.
3. Changes above five inferred pixels account for 94.26% of ETH and 81.29% of
   Hotel stationary-subset CV error. The issue is not confined to tiny changes.
4. Allowing every point a closed +/-0.501 inferred-pixel box, a native-coordinate
   constant-velocity line is incompatible with 46/59 ETH and 115/129 Hotel
   changed windows. These account for 98.30% and 97.21% of each source's
   stationary-subset native-ADE error. There were no inconclusive solver cases.
5. All 177 unchanged windows are feasible under that model. Only 27 changed
   windows are feasible. Hotel's 37 changed windows returning to the origin are
   all infeasible; they comprise seven agents/runs, not 37 independent events.
6. Every unrestricted corrected regressor has a negative seed-mean gain on the
   fixed >5-pixel changed slice. One guarded Hotel scene-neighbor tree setting
   has a small local slice gain (mean +0.1444%), but damages still rows and has
   negative gain over the full stationary subset in every seed. It is neither
   a deployable filter nor evidence that the failed full predictor now works.

## Interpretation

The results reject text serialization and the specified hidden-CV/rounding model
as explanations of most error on this subset. They do not establish true body
motion, prove acceleration, rule out annotation/footpoint shifts, or establish
that starts are inherently unpredictable. The assumed pixel boxes have not been
verified as a measurement-error model. Supplied-H projection is not a license
to claim metric or seconds-level forecasting.

Future labels appear only in these diagnostic feasibility constraints and error
slices. No recovered velocity, changed flag, magnitude bin or solver status is
an inference feature. No rows have been removed or relabeled.

## Consequence for the Next Experiment

Do not spend another training budget on precision-only repair, the unchanged
start-probability head, or another retrospective fallback threshold search.
Existing context can weakly identify some starts but does not predict their
direction and displacement well enough to improve trajectories.

The next intervention needs an explicit information/target hypothesis: verify
past-aligned body/heading context from eligible source material, or test a
prospectively registered directional motion target with additional independent
support. Static obstacle distance alone has already failed that test. More
overlapping windows at the same two sites do not solve the support problem.

The proposed new native-coordinate ADE/FDE primary protocol still awaits the
user's scientific decision. Do not change the current primary, reuse development
as confirmation, or present native-coordinate slice gains as a rescued primary
result. Independent calibration/confirmation support remains missing.

No new deployment, completed submission package, true 3D, foundation model,
metric/seconds guarantee, Stage5C execution or SMC activation is claimed.
