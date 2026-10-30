# Capped norm-balanced admission head: model and data card

## Intended use and status

Research-only gain/harm routing over a frozen seven-action trajectory bank.
Not deployable; no independent generalization, physical safety or submission
claim. This run trains cost heads, not new trajectory dynamics producers.

## Model

- Same30,405-parameter width64 Torch head and462 causal input dimensions.
- Six physical actions plus one frozen neural candidate; no new trajectory bank.
- Five predicted moments per action:benefit, harm, reference, easy reference and easy harm.
- Existing causal maximum-positive-gain decoder, all/easy predicted harm/reference<=2%, fallback otherwise.
- Same five-moment/ranking/upper anchor and dynamic admission auxiliary.
- Only change:detached `min(1e6, norm(grad A)/norm(grad U))`; zero when undefined.
- AdamW, gradient clip5, batch128, all2,000 updates per head; no score-selected early stop.
- 18 fresh heads across six contexts and seeds17/29/43;36 cached_verified upper/unit controls.

Norms are measured before clipping; the scale is not differentiated. It can
be zero or capped. This is not full GradNorm, a learned calibration method,
a new deferral theorem or proof of optimizer-step parity.

## Data and roles

149,510 unique allowed SOURCE queries from74 supplied European-source recordings
and12 localities;146,347 have some future ADE support and3,163 are wholly unknown.
The repeated897,060 context/seed row views are not independent new samples.
Eight observed steps and12 requested future steps at raw stride12; no seconds
or metric conversion. Supplied annotations are detector-silver, not human gold.

All used outcomes are exposed fitting/resubstitution labels. Reserved risk
calibration and confirmation roles remain closed. Existing producer exposure
precludes whole-pipeline OOF claims. Predictions use cached_verified past-only
inputs, frozen goals and causal velocity; future endpoints are loss/evaluation
labels only. No new test normalization, test endpoint goals or central velocity.

Known ADE averages the available requested future positions. FDE requires the
final requested position. Wholly-unknown completion sensitivity does not bound
missing interior positions in partially observed ADE labels. A required locality
lacks final endpoints, so complete-roster FDE is not estimable.

## Execution and reproducibility

Native arm64 CPU4/interop1/workers0. Checkpoint every100 steps and heartbeat
every50, explicit resume,12h runtime/10GiB disk reserve/128MiB private checkpoint
guards. Primary training36,000 updates;full refit2,000;pilot400. Training
209.66s,10.92GB peak RSS;readout125.41s;separate replay122.28s. No HPC job
was necessary or started for this measured local workload.

All36 controls match the common initialization, preprocessing, sampler/RNG
and draws. The first full new-head refit is exact. All72 inference groups and
all144 control predictions/actions/full metrics match in a separate process.
Exact replay verifies computation, not independent scientific replication.

## Results and limits

Known ADE gain0.388168%, versus unit1.904562%. Primary -1.516394pp,
locality bootstrap95%CI[-2.869731,-0.491213],3,000 resamples after context/seed
averaging. Only201/444 required recording views pass support/risk, with31 known
easy violations and72 undefined easy views. Three of six fitting gates pass.
The balanced head is not deployed; previous deployment remains unchanged.

Public:code,config,registrations,receipts,aggregate JSON bundle,actual loss traces,
gradient telemetry,figure and reports. Private:raw data,features,caches,weights
and images. Bootstrap intervals are conditional exploratory fitting intervals,
not confidence certificates for deployment or claims over independent domains.
No Stage5C execution or SMC; no true3D/foundation/metric/seconds-level claim.
