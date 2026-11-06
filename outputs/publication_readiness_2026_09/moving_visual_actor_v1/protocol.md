# Past-Image-Conditioned Trajectory Actor: Fixed Source-Held Component Test

## Material Passport

Mode: actual neural training. Previous goal turn is progress:24full cost heads,
24complete exact refits and36prediction/readout replays. Its motion component
and original risk contract fail. Better support prediction coexists with worse
conditional harm estimates. One complete12label tail case rules out missing
labels as the sole explanation. Those results remain negative and frozen.

Question:does strictly past-image motion help the trajectory actor itself,
instead of merely increasing the controller's willingness to intervene?
This is a conventional conditioning repair,not proof of a new vision method
or world-model contribution. No image benefit or deployment is presumed.

## Model Intervention

Retain the pinned deterministic EqMotion K1 author core,geometry476,existing
past-only ego/eight-neighbor support,input max-norm conditioning,source
sampler,baseline-defined loss factors,available12ADE objective and4000update
budget. Do not repeat the failed prefix-balanced objective.

Add one bias-free46->32->8SiLU image branch with zero-initialized final weights.
It adds0.1*tanh(context embedding)to the valid nodes' eight-dimensional
history-speed feature input before the original node embedding. These become
conditioned latent features,not physical speeds. Coordinates,causal velocities,
time masks and the full trajectory decoder remain unchanged. This is an
end-to-end actor,not a posthoc ordinary residual or a new rollout generator.

The46inputs are the40action-invariant image-motion/quality/history slots and
six action-invariant native-image-boundary slots from the verified causal
measurement bundle. Candidate-specific rollout fields are excluded. All six
candidate copies must agree before reducing to one row. Geometry is not
stabilized or moved into inferred metric coordinates. No future camera pose
is read,estimated from future images or supplied as input.

Compare six quality_geometry actors against six motion_geometry actors under
the exact existing six within-A source exclusions. Quality keeps reliability,
identity-suppressed history encoding and bounds;motion admits measured past
transforms. The primary is motion-minus-quality,not motion-minus-raw bundle
selection. Six original raw controls are hash-verified. One full raw-control
retraining through the new masked branch must reproduce original core weights,
optimizer state,sampler/RNG and common training trace. Zero context parameters
are inactive;nominal parameter count matches,not effective information.

All arms have3,028,996nominal parameters,1,728more than the original core.
The branch preserves the old Torch RNG after initialization;zero context
has the exact initial function and shared core initialization. Per-job
normalizers use all past fitting rows with equal-source moments,never donor
or own-B statistics. No future-validity mask enters normalization.

## Data Roles And Admission

Use the existing125540producer-A rows,14source groups,34recordings and six
source-excluding OOF designs. New modality admission applies to this actor
branch only;old image_inputs_admitted:false records stay unchanged. Existing
image/feature/split hashes are checked,not relabelled as fresh measurements.
Preparation opens past metadata and input context banks only;future targets
are separate supervised training/evaluation labels. No own-B,new C/H,reserved
Moving20,DUT or DroneCrowd outcome is opened in this experiment.

These A sources have prior exposure and rotating producer roles. OOF fitting
exclusion does not restore independent confirmation. One forecaster seed17
is a component experiment,not the three-seed formal paper result. No scene,
teacher,prototype or endpoint constructed from a donor/test future enters input.

## Fitting, Recovery And Resource Scope

Twelve full4000update treatment fits;one4000update old-control refit;two full
4000update treatment refits on the first fixed source design. The two refits
check quality and motion separately;do not claim twelve full exact refits.
Real100+100vs200motion recovery pilot is exact before full fitting. Its resumed
200prefix seeds only the corresponding first motion fit;the separate direct
200run is additional engineering,not another completed model.

Use original AdamW,lr.0003,cosine schedule/minimum ratio.01,batch64,weight
decay.0001,clip5. CPU4/inter-op1/workers0,nativearm64,no resource probing.
Atomic200update checkpoints,50update loss/heartbeat,explicit same-definition
resume,12hper-phase boundary,5GiBfree reserve and1GiBnew private artifact cap.
Healthy slow runs continue;there is no quick downgrade. Pilot estimates full
time and memory before the main matrix. CREATE queue is checked read-only;
remote observation failure is not evidence that a job is terminal or idle.
Existing checkpoints,raw data,caches and third-party source remain private.

## Frozen Evaluation And Gate

Freeze all12complete donor prediction banks before reading diagnostic labels.
Report raw,quality,motion forecasts versus causal CV with the existing
query-balanced source/recording prefix1/3/6/12statistics,unknown completion
bounds,easy/hard diagnostics and all partial/unknown rows. Raw-matched bundle
contrasts are secondary. Full12available-label ADE is not full12complete ADE.

Use the previous actor component progression screen:whole12paired lower mean
and exploratory3,000source-bootstrap CI lower positive,at least8/14known
source contrasts positive,and both producer-fold known means positive.
This only permits considering a separately registered controller experiment.
It does not replace original per-recording2%all/easy harm,signed easy-net,
reference/support/NA checks or authorize deployment. No post-readout threshold
search,cohort deletion,identity certificate or alternative primary selection.
Prediction/scalar replay must match exactly. The current compiled paper remains
unchanged until any separately documented evidence is integrated.

Native observation8/prediction12,stride1,image coordinates only. No verified
meters,seconds,homography,true3D,foundation,human-gold or physical safety claim.
Stage5C execution and SMC remain off.
