# Information-Preserving Native And Aligned Node Histories

## Material Passport

The previous completed query-view model fails its registered primary. The
one-sided follow-up is stopped before any outcome readout,not a completed
negative score. A synthetic counterexample shows that replacing native history
with aligned history can erase camera information required by native labels.
The unchanged old control remains valid. No partial checkpoint is a warm start.

Question:does per-node observed-view information add useful forecasting signal
when original native geometry is retained and future camera restoration is
not imposed?

## Fixed New Model And Capacity Control

Always retain the original native ego/neighbor history and native velocities
as the EqMotion coordinate inputs. Recompute the original causal baseline from
native history. Apply the original observed-joint-max-norm conditioning;the
scale uses the current request's past only,not fitted donor/test statistics.

Compute a second aligned view of the same valid nodes using the existing
seven-pair past affine operators. After scale conversion,this produces aligned
velocity and aligned-minus-native displacement. Concatenate native velocity16,
aligned velocity16,and alignment displacement16values per node. Use asinh
without clipping or learned normalization. A bias-free48-to-32-to-8encoder
adds0.1*tanh(offset)to valid nodes' eight pattern features. The coordinate
stream is never replaced. This is a latent feature offset,not an ordinary
trajectory residual output or a claimed error bound. Output stays in native
relative coordinates before the unchanged whole-twelve available-label ADE.
No predicted future-camera operator enters model inputs or outputs.

Two learning arms share architecture,initialization,sampler,factors,optimizer
and4,000updates:

1. native_node:suppress observed transforms to identity;the same encoder sees
   native velocity,native velocity,and zero alignment displacement.
2. camera_node:use measured past transforms for the additional aligned view.

Both have3,029,060nominal/582,468trainable parameters,1,792more than the
original core. Final encoder weights start at zero,so initial forecasts match
the old core. Preserve the original Torch RNG after encoder construction.
A frozen-encoder raw_control must reproduce the original4,000update shared
core,active optimizer state,draws,factors,RNG and common trace. Two additional
inactive optimizer parameter entries are expected,not falsely called identical
complete parameter lists. Fully refit each learning arm's first source design.

Unsupported image measurements retain the same full-window identity operators;
all rows stay. New weights can still change unsupported predictions,so this
is not a frozen-model fallback or identity safety certification.

## Data Roles And Runtime

Use the same previously exposed producer-A125540windows,14source groups,
34recordings,1613queries and six source-excluding designs. Native observation8/
prediction12at annotation stride1,image coordinates. No own-B/newC/H/reserved
outcomes,teacher exposure changes or source-statistic leakage. Cached operators,
geometry,raw checkpoints/forecasts and old camera controls are hash/row-alignment
verified,not newly extracted. Generic loading opens/hashes label banks;future
labels/masks reach only loss/evaluation,not inference. No future endpoint,
central velocity,test endpoint goals or future images/poses enter inference.

Fresh model matrix:two arms by six designs,one full raw control and two full
first-design treatment refits. One forecaster seed17only,not the formal three-
seed main result. Real recovery tests compare100+100with continuous200and a
separate200trajectory interrupted by an in-forward SIGTERM request deferred to
the committed-update boundary. Both must match complete scientific state.
The100+100prefix seeds its own first camera_node treatment. The two other200
trajectories are extra compute. Total uninterrupted budget60,400updates.

Nativearm64CPU4/inter-op1/workers0,no resource probing or multiprocessing.
Keep12hphase limit,2GiBreserve,512MiBnew-private cap,checkpoint every200and
loss/heartbeat every50. SIGTERM/SIGINT request stop at a completed update.
An unexpected partial-update exception must retain the last coherent checkpoint,
not overwrite it with an advanced sampler/partial optimizer state. Initial
step0and saved checkpoints explicitly mark a committed boundary. Same-definition
resume restores the whole state. Healthy slow training retains its full budget.
No raw/image/feature/checkpoint redistribution or deletion of old valid artifacts.

## Predeclared Readout

Freeze all twelve new prediction banks before scoring. Primary:camera_node
minus native_node,whole-twelve paired completion lower,with unchanged source/
query-balanced1/3/6/12statistics,partial/unknown accounting,3,000source draws,
seed39107. A positive result requires positive lower and CI lower,at least8/14
known-positive source contrasts and both producer folds known-positive.

Also require camera_node minus cached raw actor to satisfy that same screen
before considering any new controller. Beating a worse capacity control alone
is insufficient. Report native_node minus raw,old both-arm control,all original
six causal baselines and cached camera-CV,all sources/recordings/camera-support
strata. All raw-actor easy/hard diagnostics remain;no source or prefix is
discarded. Exact12new-prediction/all-scalar replay and both complete treatment
refits precede final reporting. The secondary cannot replace the primary.

The component screen does not certify the original per-recording2%all/easy
positive-harm,signed easy-net,reference/support/NA contract. No threshold search,
identity certificate,epsilon,empty-support pass,new controller or deployment.
Independent confirmation and three forecaster seeds remain not_run. A standard
two-view feature encoder or information-loss counterexample is not a new method
contribution by itself. No SDDt50,verified seconds,meters,homography,true3D,
foundation,human-gold or physical-safety claim. Stage5C execution and SMCoff.
