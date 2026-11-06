# Separate Past Alignment From Future Camera Restoration

## Material Passport

Previous goal turn is progress: six actual full4000update camera actors,a full
identity control,one exact full treatment refit and complete prediction/scalar
replay. Its registered component comparison fails:known-5.133066pp,paired
lower-21.447401pp,exploratory source interval[-33.682328,-9.910946]. Only3/14
known source contrasts improve and both folds are negative. All six matched
final fitting minibatches have higher loss;this is not a full training-average
loss estimate or proof that one particular operator causes the failure.

Question:does removing the fixed future-camera constraint let a past-aligned
trajectory actor improve on the original native-coordinate actor?

## Fixed Four-Arm Design

| Arm | Past histories to query view | Predicted output through future affine |
|---|---|---|
| identity,cached control | No | No |
| past_only,new full refit | Yes | No |
| future_only,new full refit | No | Yes |
| both,cached previous experiment | Yes | Yes |

Only two operator switches change. Reuse the original pinned EqMotion K1 core,
3,027,268nominal/580,676trainable parameters,nearest eight aligned complete
neighbors,geometry476,whole-twelve available-label ADE,fitting loss factors,
draws,initialization and4000update AdamW schedule. No added learned modules,
residual output,loss,early stopping or threshold search. Disabled operators
become identity matrices before the existing differentiable camera wrapper.
SourceForecaster still recomputes its causal baseline from the selected past
coordinates. No future map is concatenated as a new learned input feature.

Both enabled operators are precisely the previous implementation:seven observed
affines compose history to the query view;integer powers of the final observed
affine extrapolate future image motion. The latter is an assumption,not future
pose ground truth. If any past pair is unsupported,both cached operators remain
identity for that complete window. No rows are removed or declared safe.
Even an imputed window can change under the new trained weights.

The past_only arm predicts native-coordinate labels without a hard future
restoration. That may make camera movement unidentifiable from aligned history;
this is a possible negative result,not a reason to substitute another arm
after scoring. The future_only arm tests whether hard restoration without
past alignment also changes fit. Factorial known-label interaction is a
descriptive difference of differences,not a new deployment endpoint.

## Existing Roles And Provenance

Retain all125540previously exposed producer-A windows,14source groups,
34recordings,1613queries and the same six source-excluding designs. Native
observation8/prediction12,annotation stride1,image coordinates. Not SDDt50,
verified seconds,meters,homography,true3D,foundation or human-gold labels.
Exclusions hold within each fitting pipeline;globally rotated exposed roles
are not independent confirmation. No own-B/newC/H/reserved outcomes are read.

Cached_verified:original identity checkpoints/forecasts,the previous six
both-arm checkpoints/forecasts,original camera-CV controls and causal operator
arrays. Check their hashes and row alignment before use. Reusing them is not
fresh training or a new image measurement. Generic loading opens/hashes the
complete packed bank including labels;only past geometry and past-derived
operators enter model inference. Forecast freeze precedes scoring,not all
target-file byte access. No future endpoint,central velocity,test endpoint goal,
future image/pose or test-statistic normalization enters inference.

Fresh_run:two arms times six designs,one full identity-control refit and two
full treatment refits on the fixed first design,one per new arm. One forecaster
seed17only;not the formal three-seed main result. Real100+100vs200recovery
precedes full fits. Its own200prefix may resume the first past_only model;
continuous200is extra compute. Total uninterrupted real-data optimizer budget
is60,200updates. No other warm start. Completed receipts are immutable and
same-definition checkpoints resume model/optimizer/sampler/RNG/draws exactly.

## Primary And All Controls

Freeze all twelve new prediction banks before scoring. Primary:past_only minus
identity,whole-twelve source/query-balanced paired completion lower. Keep the
same1/3/6/12statistics,partial/unknown label accounting,3000source bootstrap
draws and seed39107. Report known differences and completion bounds together.
Progression requires positive primary lower and CI lower,at least8/14positive
known source contrasts and both producer-fold known means positive.

Secondary paired contrasts:past_only minus both,future_only minus identity,
both minus future_only,and both minus identity. Retain all four arms,all14
sources,34recordings,camera-support strata and original easy/hard diagnostics.
Also report the original six causal baselines and cached camera-CV control.
The descriptive known-label interaction is(both-future_only)-(past_only-identity)
on the same source/query support. Do not claim a sharp missing-label interaction
bound from subtracting unrelated envelopes. It is not the primary.

A favorable future_only result cannot replace the failed primary. Any later
investigation needs a separately registered question;neither arm licenses
new test access,controller fitting or deployment. The component screen is not
the original per-recording2%all/easy positive-harm,signed easy-net,reference/
support/NA contract. No identity certification,epsilon denominator,empty-support
pass,changed risk floor or selected-row deletion. Old failures remain failures.

## Runtime And Completion

Nativearm64CPU4/inter-op1/workers0,no resource probing/multiprocessing. Local
execution is feasible based on the completed previous full fits;a real pilot
measures this new run. Check CREATE queue read-only;timeouts mean unknown,not
empty or stopped. No remote duplicate submission.12hphase boundary,2GiBdisk
reserve,512MiBnew-private cap,atomic checkpoint every200updates,loss/heartbeat
every50. Save latest state on interruption where storage permits. Healthy
slow training retains the complete budget. Exact full refits and separate
prediction/scalar replay precede final reporting. Keep checkpoints/images/raw
data/private caches/third-party code outside Git. Update README/state and push
scoped code,protocols and scalar receipts after meaningful milestones.

The claim remains an exploratory model-side repair,not submission readiness or
an independently validated world-model advance. Stage5C execution and SMC off.
