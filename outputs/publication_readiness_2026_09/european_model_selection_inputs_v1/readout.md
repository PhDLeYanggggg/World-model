# Independent-Scene Input Preparation

## Material Passport

fresh_run: largest-record resource pilot, all28record input conversion and
a second complete28record raw reparse/input-array comparison.27records are
newly parsed in the full build;one pilot checkpoint is hash verified and reused.
The verification phase reparses all28,with zero cached-record substitutions.

cached_verified: the released archive, raw row identities, frozen24September
roles and existing8/12/rawstride12causal input schema. The archive is fully
hashed again in each real phase. Source training/evaluation results are not rerun.

not_run: forecast inference, future target-window construction, prediction-error
readout, model/threshold selection, normalization fitting, calibration or
confirmation. This is data engineering toward an independent-scene evaluation,
not a positive generalization or deployment result.

## Complete Cohort

|Frozen selection locality|Recordings|Target windows|Visible agent windows|
|---|---:|---:|---:|
|eu-locality-087|4|13,370|19,436|
|eu-locality-092|4|4,641|8,679|
|eu-locality-093|4|2,028|3,860|
|eu-locality-103|4|5,349|9,133|
|eu-locality-104|8|4,584|7,944|
|eu-locality-125|4|8,130|16,042|
|Total|28|38,102|65,094|

There are7,087query times and26,992short-history context rows. The original cap
is up to256query times per recording,not256guaranteed. One Brzesko recording
has only175past-supported times;no missing future, prediction error or outcome
filter explains that difference. All original28recordings remain.

The6,179,261raw rows and3,553,348past-supported target positions in the role
manifest are whole-recording support counts,not the sampled cache size or
independent sample counts. Overlapping windows and repeated seasonal recordings
do not create38,102independent scenes. Six conservative locality groups are
the intended unit for this later cross-scene model-selection study.

Queries with at least2/5/10target agents number5,240/2,857/1,281. All28recordings
have some multi-target support,but the sparsest has only3such queries out of175.
It remains in the cohort. These causal context counts justify testing joint
decisions;they do not prove a learnable gain or a future risk denominator.

## Causality And Role Checks

Every target has eight complete past observations. All currently visible agents
are retained,including those with incomplete histories. Velocity is causal
finite difference. The CV rollout is computed from those past observations;
its predicted endpoint is not a future target. No target uses track survival,
future completeness, actual easy/failure status or future endpoint goals.

The cache has only input/query-index arrays. The consumer rejects label/future
fields,wrong raw identity,wrong registration and changed files. First/middle/
last queries in every real recording give84exact comparisons against the
independent raw-prefix reader. Synthetic future-mutation/truncation checks and
the second full raw replay provide distinct evidence;the latter compares all
stored arrays,not only those84prefix cases.

The original training accessor remains unchanged and rejects selection records.
The new interface is input-only. No calibration or confirmation member is
deserialized. Original roles SHA256 remains
`858065b8a14f15a8741cbd9b0f24e5de595a02512b2552709d37b6eeb5e88435`.
Raw archive parsing necessarily reads whole released recordings,which contain
later observations;no future label window or forecast error is constructed.
This is not a claim of verified sensor-online detector tracking provenance.

## Actual Local Cost

|Phase|PID|Elapsed seconds|Peak RSS bytes|Raw parses|
|---|---:|---:|---:|---:|
|Largest-record pilot|79023|6.620355|402,259,968|1|
|All28build,one verified pilot reused|79090|15.041471|643,776,512|27|
|Complete raw reparse/array replay|79148|16.475473|820,805,632|28|

The input cache is48,651,654bytes,below the1GiBcap. The10GiBdisk reserve is
preserved;about18GiBwas available. Local execution is appropriate. No CREATE
job is submitted,duplicated,restarted or inferred from an old queue record.
Per-record checkpoints and heartbeat support explicit unchanged-input resume.
No sample budget is reduced for speed.

## Remaining Model Interface Work

This exact input schema is not yet the full current-model feature store.
The generic geometric packer expects476geometry fields and keeps complete
neighbors. The actual active producers additionally use the versioned masked-
neighbor repair,so a bridge must apply that repair rather than use generic
packing alone. Retaining every visible input agent also does not mean the
network consumes more than its eight nearest masked neighbors. Current detector-aware
models also need64past confidence/box-change fields. The original input cache
does not store confidence history,so those fields require a raw-prefix-aligned
extraction;inventing confidence or aliasing new rows to old SOURCE IDs is invalid.
Static scene-region context must likewise be explicitly available or marked
missing,not filled from future trajectories.

Next: build that causal geometry/detector bridge,bind the actual producer and
normalizer lineage,freeze predictions and comparison rules,then perform a
versioned model-selection readout. The appropriate existing active producer
family has three actual forecaster seeds;the previous three cost-head seeds
sharing forecaster43 are not a substitute. Risk calibration and final confirmation
remain separate and closed. This input registration does not authorize errors.

## Scope

EuropeanSquares released oblique detector-silver/local-image coordinates,
8observed/12requested/stride12rawframes. Not SDDt+50,metric,seconds,human gold,
true3D,foundation,physical safety or submission-ready evidence. No training,
deployment promotion,Stage5C execution or SMC.
