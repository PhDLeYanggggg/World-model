# Real-Experiment Continuation Handoff

## Source-Site Diagnostic Live (2026-09-18)

Goal active. New source-site information diagnostic running. PID8914,
session63363,private source_site_probe_v1/run.log andheartbeat. Do notstart
a second writer or modify frozen training code/config. Registrationpusheda91e5795.
Config `configs/m3w_source_site_probe_v1.json`, SHA
`a2038307eab36cc62ae783eedf3f359747965f2ddcc17cf4f661d83273fab74d`.
Thirty models,2arms,5physical source sites,3seeds,60kfixedupdates. Same model,
inputs and label as preceding visual probe;only source internal held-site test.
Support audit fresh:22,374complete rows/726IDs/36videos/5sites,bothclasses each.
Original SDD train40 only. No main fitting/evaluation,sealedroles or formal
protocol changes.42focused tests pass;fulllegacy suite notrerun.
Pilot100updates/4.49sec completedwithoutheldscore;fullrunresumeswithinbudget.
After fullwriterexits:exactpredictionreplay,pairedsamplingverificationandzero
updatecompletedresume,thenconditionalblockanalysis. Newanalysis/verifier are
unboundpost-fit consumers,nottrainingchanges. Priorvisualmatrixcomplete.
UnrelatedstagedSHA stays
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

Supplementaryannotationauditcompletedwithoutfilter/trainingchange.10,039positive
labels;8,583maxtravel<0.1currentboxdiagonal;median2.06annotationpixels;244>=0.5
currentboxdiagonal. Currentboxdenominator differs fromoldermedianpastboxdefinition.
All22,374historyjoins exact;99.686%sampledpastannotationrowsgenerated. This is
offlineannotations,notstrictsensorasof ormotionintentiongold. The one-pixel bin
uses1e-6roundofftolerance(float32restoration);2,248positives<=1pixel,not165rawfloat
comparisoncount. No inferenceinputs contain labels.Qualityaudit3targetedtests.

## Matched Visual Probe Complete (2026-09-18)

Goal remains active, not complete or blocked. All30 registered fits/36prediction
cells/60,000 updates completed; 100 pilot plus59,900 full continuation. Full wall
47.114min, summed fit2693.539sec. No new forecast, threshold selection or model
promotion. RegistrationSHA9edee7d4fe9ef0b97d370ad7804f78ec094102580ef5e7ba12e8e0adf5596af9;
reportSHA91914580ba5bad24c66b87878ad0d8258253356c2e62151c779807b3e2f0bc7a.

Mixed RGB vs mask has positive row BrierliftETH/Hotel .039378/.036045 in allseeds,
but equal-agent lift -.019117/-.038780 and both conditionalCIs cross0. NoRGB
schedule beats its own constant training prior in both sites. Source-onlyRGB
worsensETHallseeds. Primary row contrast stays positive;do not replace its
estimand or claim a robust contribution from it. ExactBrierdecompositions and
single-agent-omission sensitivity are in analysis.json/results.md. No held-label
recalibration was performed. Mainonlyfits overfit; finite source budget not
claimed converged. Pixels vary;source box median10.9x12.7modelpixels,31.5% axis<8.

Thirty exact probabilityreplays,15matchedstreams,91immutableartifacts unchanged
oncompletedresume with0newupdates.34focusedtests passed;fulllegacy suite notrerun.
PNGfigure inspected after tickspacingfix. No rawimages/cache/weights committed.
Run log and private assets: data/stage_cvpr2027_experiments/source_visual_start_v1.
Public conclusions/gates/repro and allaggregate metrics under matchingreportdir.

Fulltrainer PID3037/session47028, replay7255/session42040, verification50217,
analysis44379/2196, audits and tests are all terminal exit0. NO CURRENT LIVE
TRAINING from this experiment;do notpollthesehandles orrestartcompletedfits.
Earlier live notes below are historical. Newcodecommitbeforetrainingdb7d30f4;
intermediate audit/statuspush0c70433e. Finalcompletioncommit comesafterthisnote.

Next useful action: audit class/agent support by the five admitted physicalSDD
sites,then register a source-site-held-out RGB/mask information test to distinguish
failure inside source from source-to-main transfer. Only originaltrain40;do not
openmain development/calibration/confirmation. This remains exposedfitdiagnostic,
not newindependentconfirmation. Do not repeatmainthresholds,changeprimary or
promotea residual from classificationgain. Largerbudget/label/visibility repairs
need controlled comparisons,not a claim that currentpixels carry no information.

Main11966forecastqueries/native8to12/equal-sitepast-normalizedADEunchanged.
Currentcausalwording is offline supplied annotation histories,not strictsensorasof.
NoStage5C/SMC/metricseconds/true3D/foundation/submissionreadiness claim.
Unrelatedstagedfingerprintmustremain
c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323.

## Matched Visual Probe Live (2026-09-18)

Goal active. Current trainer PID3037 / session47028 is LIVE. Do not start a second
writer or modify registered code. Registration `m3w_source_visual_start_v1.json`,
SHA `9edee7d4fe9ef0b97d370ad7804f78ec094102580ef5e7ba12e8e0adf5596af9`,
pushed before the full run at `db7d30f4`. Thirty models /36 prediction cells /
60,000 fixed updates; paired mask-only and past-RGB arms, three schedules/seeds.
Pilot PID2901 ended exit0 at100updates,4.348sec,no held evaluation. Its checkpoint
is resumed in the full matrix. Current progress/heartbeat and log are under
`data/stage_cvpr2027_experiments/source_visual_start_v1/`.

Thirty-two focused tests pass. Fresh input pixel audit is complete: all365 main
and22,374 source histories have temporal pixel change; no absent common adjacent
pair. This is not visible-intent evidence. Pixel audit does not filter rows or
change labels, thresholds or training. Existing input hashes/roles preserved.

After training exits, run exact probability replay, the dedicated verifier,
then fixed analysis. Verifier expects30fits/60ksteps,15matchedstreams and91
unchanged immutable artifacts on completed resume. See local reproduction guide.
All post-training checks remain pending at this snapshot. No new deployment,
forecast lift or independent confirmation. No Stage5C/SMC.

Earlier no-live-training notes below refer to previous completed experiments,
not this current visual matrix. Unrelated staged fingerprint is still
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

## Source Start-Information Comparison Complete (2026-09-18)

Goal active,not complete or blocked. This turn tested the next hypothesis after
the failed conditioning study: admitted SDD source plus explicit stationary-start
classification. All45 fixed models completed(15logistic,15trees,15Torch MLP),
54prediction cells,15,000Torchupdates. Source-only fits shared across two held
directions,not54independentfits. No forecast or threshold was changed.

Registration`configs/m3w_source_start_probe_v1.json`;
SHA`bbe433f82cbbe40f393cff61b15a7ddfd9178fd95c4895eceea4a06fe9de7cd2`.
Pre-fit pushedcommit`f0d51451`. ReportSHA
`57a82444602b534601138364153ecd107db6d5036bf2cf3431d591661a04879b`.
Private`data/stage_cvpr2027_experiments/source_start_probe_v1`;
public`outputs/publication_readiness_2026_09/source_start_probe_v1`.

Source22,374complete stationary rows,726IDs,36videos,5sites;6,460incomplete
stationary queries unscored. Main365windows,31IDs(ETH81/5,Hotel284/26),Zara no
stationary queries. Source labels are any annotation-center change,not intention.
Source+144rawframes and mainnative horizons not physically equated. Entire main
11,966forecast cohort and primary equal-site past-normalized ADE unchanged.

Source-only BrierliftETH/Hotel:logistic-.071974/+.068868,trees-.110321/+.066763,
MLP-.119195/+.058992. Mixed also worsens ETH in allfamilies. No bidirectional
positive arm. Source/mixed Hotel improvements versus mainprior all lose to the
constantSDDprior(Brier.247935);exact score decomposition shows positive mean
shift but negative within-site-varying-probability value. This is descriptive,
not causal identification or held-label recalibration. Main-only neural fits
overfit. Every source/mixed Hotel conditional agentCI versus prior crosses0.

No fit warnings;684.696summedfitseconds,11.61minfullwall. 100pilotupdates+14,900
continuation. 45exact probabilityreplays;166immutableartifacthashes unchanged on
completedresume,0fits/updates.17focusedtests passed;fulllegacysuite notrerun.
All current pilot/training/replay/verification/analysis sessions ended exit0.
No live training to wait for;do not poll oldhandles. Imagepreview visuallychecked.

Next: a fixed source-supported visual-state information probe,comparing matched
geometry+coverage-mask against actual pastRGB and constantprior controls. Preserve
source admission and mainroles;do not redo generic geometric thresholds or claim
the probability gain repairs trajectories. Check annotation label resolution,
input coverage and source clock caveats. No residual/policy promotion until
information and trajectoryutility are established. Independent confirmation
remains unresolved;do not reopen it for this repair.

All raw/cache/modelweights stayprivate;sourceprobeprivate~1.1GB/public~0.5MB.
Prior tests initially exposed only an exact-float assertion;fixedbeforefit.
Unrelated staged fingerprint remains
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
Other staged/usergenerateddata-lake changes untouched. Historical notes below
are not currentpendingtasks. No Stage5C,SMC,metricseconds or modelpromotion.

## Unit Conditioning Complete (2026-09-18)

Goal active, not complete or blocked. This turn measured actual training-role
parameter gradients, repaired a native-unit cutoff in rollout features, and
completed 27 fixed geometry-only fits (162,000 updates) plus nine verified old
controls. Do not repeat these as fresh work. New multimodal contribution remains
unproved; main metric/cohort and all sealed roles stay unchanged.

Registrations: normalization_response_v2 and m3w_unit_frame_training_v1.
Training registration SHA663cb62a13281fb3533e95d28b23814452560292e54cbcb428ca1e0dc6bbda7e.
Training report SHA f5a0c5a56b3e4c2f131f214745bfe252e94ca74209ec98c1be7becb990d56954.
Old gradient v1 is immutable and bound by v2; do not overwrite it. v1 frame's
tiny-curved unit-rescaling failure is repaired in the separate v2 frame module.

Actual gradient audit: 432 v1 records,216 v2 records,zero optimizer updates.
Main fold0 training only; up to32 complete/anchored windows per event/domain,
three seeds/modalities. Source legacy final static-start norm0.003087 vs
other-motion1.449280. These are sampled training gradients, not predictive lift.

New fit gains vsCV: inputs-only -0.68730%, radius decoder -2.48928%, internal
loss -185.77715%; old control -0.80523%. 3 local Hotel fits positive,0/27 easy
pass. Internal loss avoids clipping but causes large absolute harm. It changes
training objective only; primary remains equal-site past-normalized ADE.
Post hoc future-label expanded-pool oracle2.63722%, NOT a learned selector.
No-anchor guard-only effect on legacy is tiny (-0.80523 to-0.80027%).

All27 exact prediction replays;82 unchanged artifacts on completed resume,
zero updates; sample counts/final RNG match parents. 24 focused tests passed.
Summed fit277.02sec, full invocation about290.1sec; local CPU4 was adequate.
Pilot PID96284/session96695, trainer96476/session12946,
replay97023/session23330, verification/session62000 and tests/session75734
all terminal exit0. No live training remains. Do not poll old handles.

Next action: separate useful causal cues from decoder-induced harm on training
roles, with a matched registered follow-up only if justified. Do not enlarge
the failed internal-loss recipe, repeat held-threshold tuning, or reopen sealed
roles. Main challenge is identifiable start/direction and safe selective benefit,
not another runtime repair. Independent confirmation remains separate/unresolved.
No model promotion, restored historical Stage37 claim, Stage5C, SMC or readiness.

Reports: unit_frame_training_v1/{report,contrasts,failure_analysis,gates,
reproducibility,report.json,analysis.json,verification.json,replay.json,
oracle_diagnostic.json,fit_metrics.csv,loss_trace.csv}. Raw caches/weights private.
Unrelated staged fingerprint remains
c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323.
Earlier pending and running notes below are historical, not current.

## Mechanism Controls Complete (2026-09-18)

Current goal active, not complete or blocked. Previous turn completed the first
SDD auxiliary matrix; this turn completes the matched mechanism controls.
Full trainer PID80360/session81910, replay93083/session51950, checkpoint stream
check/session97527 and resume/session28672 all observed terminal exit0.
Do not poll old handles or start duplicate fits. No live training remains.

54 fresh fits,270,000steps,8,251.3883summedfitseconds; fullwallabout2.33h.
54 parent fits cached_verified, not freshly trained. Exact new replays54/54;
completed resume166immutablehashesunchanged0updates; sample streams match.
23focusedtests pass; fulllegacysuite notrerun. Main11,966fitrows/primary/closed
roles unchanged. ReportSHA cdbbcece43a678577b97fa6f1d5d702026323b63d751965f50d0dfcccf7780b4.

Every aggregate losesCV; 3/54newfits tinypositive but 0/54easypass,0/108safepositive.
Real source vs permutation: geometry+.11789%,mask-.01536%,RGB+.05827%; allsiteCIs
crosszero. Main4k beatsmain6k inall9input/site seedaverages. Source advantage
ispartlymainexposure; correct source pairing no stable contribution established.
Do not conclude labels have zero information or use permutation as fullindependence.
Source stationaryscale diagnostic and event counts are in source_support.json.

Next justified repair: stationary/near-stationary internal source representation
andlossresponse. First test syntheticcoordinate-unitrescaling invariance and
measure actualeventconditionedgradient contributions; analyticdlog1p is not
parametergradient. Thenregisteronematchedtrainingcontrolkeepingmainprimary,
cohortandclosedrolesfixed. Do not silentlychangeprimary, repeatthresholdsweep,
or promotehistoricalcontaminatedStage37results. Independentconfirmation remains
separateandunresolved. NoStage5C/SMC/deployment/submissionreadiness.

New scripts: analyze_m3w_auxiliary_mechanism.py,
verify_m3w_auxiliary_mechanism_run.py, verify_m3w_auxiliary_mechanism_resume.py.
Reports: sdd_auxiliary_mechanism_v1/{report,analysis,contrasts,failure_analysis,
reproducibility,replay,stream_verification,resume_verification,fit_metrics}.
Frozen training code/config unchanged; analysis tests separate.
Unrelated3019stagedentries must stay untouched; stagedfingerprint
c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323.
Older live/pending entries below are historical and superseded.

## Source Mechanism Controls Registered (2026-09-18)

Update: registration committed/pushed as bba7fbe1; SHA256
699c9f6fdb346b828dee4411894b2747a05f541e404285325fa0a93f2174837a.
Pilot PID80309/session85879 observed exit0, 100 real updates / 4.444 seconds,
no held scores. FULL TRAINER PID80360 / SESSION81910 IS LIVE; poll this handle,
do not launch duplicate or change registered training. Goal active.
Source diagnostic runs PID handles99852/89506/97254 terminal exit0.
New source_support evidence: broad static-to-any-motion10,039windows342tracks;
stricter half-box244windows58tracks reconciles exactly with train40 old census.
Static normalized target medianSDD1125 vs mainfold0training24.702; analytic
dlog1p/dADE median.000888 vs .038907. Native-unit.001floor is not stationary
scale invariant. These are diagnostics, no training or primary changes.

Prior turn was progress: full 54-fit auxiliary experiment and exact replays,
not a successful method. Next fixed experiment separates fewer main updates
from useful source supervision. Config: configs/m3w_sdd_auxiliary_mechanism_v1.json.
New main4k and sdd_permuted arms, three modalities/seeds/folds, 54 fresh fits,
270,000 updates; old 54 controls cached_verified only. Source permutation moves
baseline-relative loss labels within original recording and exact future-support
stratum, never inference features; singletons/dependencies must be reported.
No main task, split, admitted source or closed-role changes.

Twenty targeted tests pass, including exact zero-pretraining resume and matched
main sampler. Previous real CPU cost supports local 2-3 hour estimate; no new
HPC query or job, no need to transfer 2.61 GB cache. Freeze/commit before pilot,
run 100 updates of sdd_permuted_past_rgb_seed17_fold0 without held scores, then
resume the same fit as part of the full matrix. Do not claim completion before
terminal status, all trials, exact replays and completed-resume checks.
Goal remains active. Independent confirmation unresolved. No deployment or Stage5C/SMC.

## Auxiliary Comparison Complete, No Safe Gain (2026-09-18)

Registration SHA256: 17d12a071c73a44a42c42c13341c18f4cc4093188b719a4c6a0b57623e8e5fbb.
All 54 fixed fits complete, 324,000 real Torch updates, 10,660.60 summed fit
seconds (about three hours wall time). All 54 checkpoint replays exact.
Completed resume preserves 163 artifact hashes and adds zero updates.
Trainer 61978/session74306, replay 78345/session55332 and resume
78806/session78029 all observed terminal exit 0. No training process remains
live; do not poll old handles or rerun completed fits as new experiments.

Full source inputs: 229,333 queries, 254,841 crops, 2.61 GB. Independent 120
frame/944 crop/120 geometry-label replays exact; all query joins verified.
Each source fit draws 128,000 samples, 97,892-98,207 unique, not a full epoch.
No original SDD val/test raw inputs used. Main 11,966 rows and closed roles unchanged.

SDD gains vs CV: geometry -0.80523%, mask -0.81626%, RGB -1.27335%.
SDD versus matched no-SDD neural controls: +0.53728%, +0.54864%, +0.69552%.
All 54 individual held fits remain negative and fail easy <=2%; train gains
are positive. RGB does not beat its mask control overall. Three reused physical
sites/three seeds/2,000 descriptive bootstrap draws are not confirmation.
No model promotion, Stage5C or SMC. Not submission-ready; full goal still active.

Analysis-only zero-reference division repaired: static-stays gain undefined,
absolute harm retained. Training bindings unchanged. Thirty-six focused tests;
full legacy suite not rerun. Reports under sdd_auxiliary_v1 include conclusions,
failure_analysis, reproducibility, report/analysis/replay and fit_metrics.csv.

Next useful question: distinguish useful source learning from fewer main-domain
updates/regularization, and measure missing transferable start/direction cues.
The current source contrast does not isolate these mechanisms. Do not launch
another threshold sweep over negative predictions or reopen sealed labels.
Independent-confirmation assets/protocol remain a separate unresolved need.
Older pending/running entries below are historical, superseded by this completion.
Prior unrelated staged fingerprint:
c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323.

## SDD Auxiliary Admission Approved (2026-09-18)

The user explicitly replied "按这个方案继续". The source-role/sampling blocker
is resolved: original train-40, stride12, obs8/pred12, matched auxiliary/control.
Do not ask again or treat the older pending sections as current.
Registration: configs/m3w_sdd_auxiliary_v1.json. 54 new fits, 324,000 updates;
full229,333pastwindows need a new complete crop cache. Main primary and sealed
roles unchanged. Thirty-four targeted tests pass; training not started at this
registration. Input pilot15,809queries/16,663crops complete42.41seconds; three
frames/36crops and geometry/labels replay exactly. Full preparation running;
verify full inputs, run100step training pilot without held scores,
resume full matrix, then exact replay and completed-resume checks. Goal active,
not complete. This experiment remains exposed-fit exploratory, no deployment.

## Resumed Goal: Manuscript Evidence Repair (2026-09-18)

Current goal remains active. The preceding resumed interaction was clarification,
not new experiment progress or a verified process wait. The user asked what the
scientific choice meant; the answer recommended original SDD train-40, raw stride
12, obs8/pred12 as a separate auxiliary arm. That question is not acceptance.
No new approval was found. Do not repeat the optional question or infer a reply.

This pass makes a scoped manuscript repair rather than starting unauthorized
training or repeating unchanged runtime checks. The motion-start result was
already in the body; its abstract lagged. The SDD geometry/image bridge was
missing from the manuscript, and the independent-protocol page still described
an old development process as running. Both are corrected. A claim table and
explicit role/observation paragraph separate pipeline evidence, probability
signal, forecasting gains and independent confirmation. No main rules change.

Four aggregate hashes and per-record totals were verified; 54 local links resolve.
See `manuscript_evidence_reconciliation.md`. No fresh model fitting, replay,
bootstrap, raw-data recheck, full tests, external-paper review or CREATE query.
No process was launched. Last SDD extraction/check handles remain terminal; do
not poll them. No new deployment or submission-readiness claim.

Next consequential step is still a matched auxiliary/control experiment after
the source-role/sampling decision. The geometry index and sampled image cache
have different coverage; do not claim the latter serves all229,333stride12rows.
The repeated-blocker audit restarted when the goal was resumed; the old blocked
snapshot below cannot alone justify immediately blocking this resumed goal.
Do not manufacture another same-site fit grid or repeated document updates while
waiting. Stage5C/SMC remain disabled. Full goal is not achieved.

## SDD Multimodal Join Complete, Repeated Scientific Decision Blocker (2026-09-17)

Previous goal turn was progress (geometry bridge); this turn also completes a
real prerequisite, not another fit-only sweep. The original 40 train recordings
now support the fixed sampled geometry queries with eight past ego-image crops.
The old image store covered only64frames/video. Pre-execution commit74ff40c1.
5,074joins,39,144unique frame/agent requests,38,449afterframe63;362,417sequential
decoded frames,30,798requestedRGBframes. All histories have some retained pixels
at each step, not all pixels/people visible.3,721partial crops,84inferredborder.
Private arrays323,957,560bytes;755.11s sum build/check durations,not training.

Actual CNN forward interface verified, zero optimizer updates, no predictive
gain/non-collapse/training claim. Independent decode120frames/143crops exact.
Completed resume preserves481artifacthashes and performszeroextractions.
29focusedtests pass;fulllegacy suite notrerun. Limited assistant self-review of
bookstore0/deathCircle2/hyang7 contact sheets, not human gold. Projected short
bbox axis<8px in15,145/39,144crops; this is descriptive, not a causal explanation
of prior ETH/UCY failures or a new exclusion rule. All registered queries kept.

Public sdd_multimodal_bridge/:report,verification,input_quality,resume,visual_review,
conclusions. Private cache underdata/stage_cvpr2027_experiments/sdd_multimodal_bridge.
Pilot30612/full53576/verify12040/analysis51034,82883/resume2355/test52107 all
observed exit0. No process remains live. `ps` was denied; it does not establish
a Torch failure. Direct tool handles plus advancing receipts/frames were used.
No new CREATE query/job. Config8bce6b4397b854e31ba40b2500eb3a003d0316cdaea755eec634069904bbf5b9;
report0baf077c4c1b67ffa7d645fa8cdb4757c44d86a797457473625e310216c7f7d8.

Blocked audit: the same auxiliary source-role/sampling condition is recorded in
the motion_start_information completion, sdd_step_bridge completion and this
completion (also earlier preceding repairs). No answer to the focused question
has arrived. The independent identified data prerequisites are now complete.
The next comparative training needs that choice; there is no live run to wait
for, and another two-site threshold/classifier grid would not advance the goal.
Do not repeat runtime/HPC checks or manufacture further diagnostics solely to
avoid the decision. Do not infer a preselected option as a reply. Recommendation:
separate original-train SDD auxiliary arm, stride12rawframes, matched control;
keep main8/12,primary,11,966fitrows and sealed roles fixed. Goal not achieved.
Once answered, register actual training roles/budget/seeds and matched comparison.
Stage5C/SMC remain disabled. No deployment or metric/seconds/true3D claim.

## SDD Geometry Bridge Complete, Training Contract Pending (2026-09-17)

Meaningful independent progress, not a new classifier sweep or model result.
Checked whether SDD admission was an artificial permission blocker: prior SDD use
is authorized, but the new auxiliary sampling/source-role experiment is not
specified by the frozen main contract. Sent one focused async question this turn:
separate stride12 auxiliary training (recommended), stride1 auxiliary only, or
no auxiliary arm. No reply observed; do not repeat it or infer an answer.

Pre-execution commit `eeb0628e` fixes bridge code/registration/check contract.
`SDDStepAdapter` reuses the 476-column geometry/8-to-12 interface, index from past
support only, separate masked labels, no old cached velocities/time/teacher.
Original 40 train videos / 8,005,367 raw rows in five scenes. Stride1 indexes
3,045,974 queries, stride12 229,333; not independent observations. Neighbor types
can be mixed. Offline generated/occluded provenance retained, no sensor-as-of.

5,074 untrained geometry Torch forwards exact CV; 320 real mutation/truncation
checks pass. Separate contiguous-run verifier checks all 3,275,307 index rows.
Index total 118.72 MiB, 17.36s summed conversion/checking. No optimizer updates,
no new forecasts/accuracy results/full image access. Sample missing labels are
reported separately, not full-population availability. Main primary unchanged.
Completed resume: zero new records, all 121 index/receipt/report hashes unchanged.
Test commands16 and25 pass with5 shared; full legacy suite not rerun.

Private `data/stage_cvpr2027_experiments/sdd_step_bridge/`; public reports under
`outputs/publication_readiness_2026_09/sdd_step_bridge/`. Pilot7471, full39302,
tests75959/31832, resume17280e(command chunk, no session), verifier99556 all
terminal exit0. Last resume heartbeat PID35456, complete. No process live.
Registration SHA cf5f6834cc5973612c2764d8de58fca7bfade6957e8a4d1d6831a24fd1f1369e;
report SHA1970c2a7feacf3f6bfa41ebed507374e05edf145ab9c47dc4cb4c51c1f22faa8.

Next: once the scientific choice arrives, register a matched auxiliary/control
study, fixed sources/updates/seeds, no sealed role access. If image cues are used,
connect verified per-query image support rather than claiming the geometry-only
bridge is multimodal. Do not redefine the complete-label primary using partial
auxiliary labels. Do not manufacture another axis/loss/classifier grid while the
choice is pending. CREATE blocker unchanged, no remote query/job. Goal remains
active, not submission-ready. No deployment, Stage5C, SMC or metric/seconds claim.

## Motion-to-Start Information Complete, One Direction Only (2026-09-17)

Progress: direct probability supervision distinguishes localized information from
bidirectional forecasting failure. Registration/code committed before fits as
`82c15be0`. Four nested past inputs (13/48/76/118 cols), logistic and ExtraTrees,
three seeds, ETH/Hotel held sites: 48 fresh sklearn fits, 2.55867 summed fit seconds.
No Torch/trajectory training. No warnings. No new primary or role admission.

ExtraTrees magnitude/directed Hotel->ETH AUROC .81048/.82152, Brier lift versus
training prior +.08030/+.07870. ETH->Hotel AUROC .50038/.51181, Brier lift
-.00484/-.00101. No setting has positive mean Brier in both directions; logistic
all fails. Tree added-motion versus quality conditional held-agent CI crosses zero.
Five ETH and 26 Hotel IDs; 365 overlapping rows, 45 runs. Zara static support=0,
not_run. No fresh confirmation or trajectory contribution follows.

48 exact classifier prediction replays; completed resume zero fits and 97 hashes
unchanged. 18 focused tests pass in 1.67s; full legacy suite not rerun. Aggregate
plot viewed; no visual issues. Fits72928, replay20887, tests14860, resume10223 and
analysis20990 all observed exit0. Resume heartbeat PID32297 state complete;
no live process remains. Private models/predictions under ignored data, not Git.

Registration SHA256 `0496ef32db42a9db1856061c2a8476cba2acf58cfc536dd64a2b2478e37d1849`;
report SHA256 `c1d377b37aa85931078e8439ddf6ccfcc0875a52a0beb0d27ff80dbb74d335b6`.
Public `motion_start_information/`: all outcomes, conditional group intervals,
replay/resume, aggregate SVG and conclusions. No full-study success claim.

Do not fit a residual or threshold from the favorable direction. Next useful
work needs independently supported observable state changes, not repeated loss/
axis/probability grids on the same exposed sites. Pending SDD auxiliary-role
decision remains unanswered; do not repeat the question or silently admit it.
If no source is admitted, record the support blocker rather than manufacture
progress by another two-site sweep. CREATE access blocker unchanged, no new
remote/runtime probe. 8/12 task, all 11,966 forecasting fit rows, approved primary
and closed Students/development/calibration/confirmation unchanged. Goal active;
no deployment, Stage5C or SMC.

## Past-Frame Controls Complete, Forecasting Negative (2026-09-17)

Progress: matched causal-coordinate training isolates static direction ambiguity
from no-anchor fallback. Registration/code pushed before training as `a91f886a`.
36 fresh fits plus 18 cached-verified controls; 144,000 new optimizer updates,
114.5830 summed fitting seconds. This is a controlled small experiment, not full
training. CPU4/interop1/workers0; pilot400 resumed into the fixed 4,000-step fit.

Past-frame primary gains: quality -0.86256%, directed -0.88781%; all 36 new fits
remain negative and easy preservation fails. Quarter-turn prediction gaps become
zero in the tested static histories. This is coordinate consistency, not accuracy.
Moving source inputs were already heading-aligned: arbitrary rotated-feature
stress cannot establish an original moving-source pipeline bug. Scope tests added.
Anchor counts: ego 11,601 / neighbor motion 354 / neighbor position 3 / none 8.
No-anchor CV guard is shared with a separately trained guard-only control.

54 checkpoints replay bit-exact; completed resume preserves 109 hashes with zero
updates. Twenty-four focused tests pass; full legacy suite not rerun. Analysis SVG
visually checked. Public `past_frame/` contains aggregate reports, replay, resume,
coordinate diagnostics and conclusions; arrays and weights remain private.
Pilot87380/PID28725, train11155/PID28809, replay23363/PID29243, tests87228,
analysis40827 and resume99190/PID29485 all observed terminal exit0.
No active process remains. A locale-only shasum failure was resolved with LC_ALL=C;
it was not a Torch/runtime failure.

Registration SHA256 `5e60a26479af017f583c476beea461785c765daeb11b57043c3120c167d2ec2d`;
report SHA256 `d4fd80d1d23ad11f1de672225e8bb278482f714d7491b9e9be8b9a8dabf64f94`.
Next: independent state-change cues/support, not more axes or thresholds on the
same exposed folds. SDD auxiliary-role decision remains unanswered: do not repeat
the question or silently admit it. All 11,966 fit rows, approved 8/12 primary and
closed Students/development/calibration/confirmation remain unchanged. CREATE
access blocker unchanged; no new probe/job. No deployment, Stage5C or SMC.
Goal active, not submission-ready.

## Residual Range Training Complete, Negative (2026-09-17)

This turn is progress: a matched 2x2 neural experiment distinguishes poor fit
from failed transfer. Registration/code/tests pushed before fitting as `a68c9342`.
Goal active, not submission-ready. No unchanged HPC/runtime probing.

`configs/m3w_residual_range.json` fixes linear/sinh readouts crossed with
log1p-ADE/asinh-residual SmoothL1. Same two feature variants, three seeds and
three fit-scene folds, 4,000 row-uniform updates, fixed final checkpoints.
54 new fits =216,000 updates; 18 original linear/log controls are reused with
exact replay. Actual summed fit time 196.2571 seconds, not a long/full training.
Pilot400 resumed without held evaluation. CPU4/interop1/workers0 remains stable.

Directed held primary gains: -0.97244/-0.90994/-15.79533/-171.03448 percent.
Transformed loss improves training diagnostics but not transfer. All54 fresh
fits are negative and easy fails. Sinh/asinh quality hits11 training coordinate
caps and1 held; directed has0 training,7 held. These numerical guards are not
physical constraints. Native diagnostics per recording remain negative.
Do not claim mere output-range optimization solves the startup bottleneck.

Public `residual_range/` includes metrics, losses, per-recording/slice diagnostics,
exact replay, resume check, aggregate SVG and conclusions. No arrays/weights.
All72 checkpoint predictions exact on replay; completed resume changes zero
of145 checkpoint/prediction/report hashes and performs zero updates. Twenty-two
focused tests pass; full legacy suite not rerun. Plot visually inspected.
Training10898/PID25333, pilot54105, replay24395, analysis34995 and resume35453
all observed terminal exit0; no active process remains.
Registration SHA256 `19f6bf3a6f1232ae59465f87ee7711b26925e83b95d4dca28eb58816fd030ab1`;
report SHA256 `d4c4e83b4dcc8c5c8cd66977150fe9dd498de50845d7d066af8479315fa2fbc7`.

Next: independently supported state-change information, not another output-range
or threshold grid on the same exposed folds. The current data are insufficient
to claim what new cue will succeed. Pending SDD auxiliary-role admission remains
unanswered; do not repeat the question or silently train it. 8/12 task, approved
primary, all11,966fitrows and closed Students/development/calibration/confirmation
remain. No new CREATE job, deployment, metric/seconds, Stage5C or SMC.

## Frozen Candidate Ceiling Complete (2026-09-17)

This turn is progress: a numerical action-class diagnostic changes the next
experiment priority. The goal remains active, not submission-ready. Registration
and tested implementation pushed as `46b664ce`; no model is trained this turn.

The preceding sampling study completed 54 real fits plus 18 exactly replayed
controls (`3eb16790`). All fresh fits failed positive complete-cohort primary
gain plus easy preservation. The current diagnostic evaluates those frozen
forecasts: 72 oracle computations, exact full recomputation, and completed resume
with zero new oracle minimizations/model updates. All inputs/checkpoints retain
their original hashes and exposure roles. Public evidence is in
`candidate_headroom/`; per-row labels and oracle alphas remain private.

Perfect binary selection over eight candidates per seed gains 1.62653%; perfect
whole-path scalar correction gains 1.72618%. Future labels supply these choices:
neither is a learned result. This limits the union of baseline-to-candidate
segments, not arbitrary mixtures, waypoint corrections or new predictors.
Independent SciPy minimization of 864 real paths agrees within 1.777e-15;
25 focused tests pass. The full legacy suite is not rerun. The registered
primary remains past-normalized ADE with equal physical-scene aggregation.

Exactly-static histories are 365/11,966 fit windows but contribute 89.3667% of
equal-scene normalized CV error. The 188 static-to-movement rows have only
0.19903--0.43404% conditional scalar-oracle gain across seeds. Improving candidate
onset/direction information is more useful than another router over this pool.
Do not silently switch the primary to favorable moving histories.

No process remains active. Registration digest is
`2507b85fb683796ffd80457450534392fbab56c8b0432ef89a91b76aa79de75b`;
main report digest is
`a393fbd6f58c3e2246b3ebd0b6b643c69bd9f9e8a12783b985c1972a01417862`.
Use `run_m3w_candidate_headroom.py`, its `--verify` mode, the independent
`verify_m3w_candidate_headroom.py`, and `plot_m3w_candidate_headroom.py` in scripts.

README/results, working manuscript and deferral prior-work status are updated.
Official CVPR dates reconfirmed: registration Nov 10, paper Nov 16, supplement
Nov 23, 2026, AoE. The 2027 AuthorGuidelines endpoint still returns 404; CFP says
LLM details remain under finalization. Do not import prior-year rules as current.
Pending auxiliary SDD source-role admission is still unanswered: do not repeat
the question or silently train on SDD. Students/development/calibration/
confirmation remain closed. No new CREATE query/job, deployment, Stage5C or SMC.
Preserve all unrelated staged files using explicit-path commits only.

## SDD Event Support Census Complete (2026-09-17)

This turn is progress: full source-level count changes the auxiliary-data
recommendation; it is not another status-only wait or a forecasting improvement.
Prior input-reader work pushed as `5d614580`. Goal active, not submission-ready.

New `m3w_sdd_state_support.py` separates eight-step past eligibility/features
from twelve-step future event labels. Fixed raw-frame strides 1/6/12/30 are
diagnostics, not selected training intervals or physical time. All 60 source
files reprocessed: 10,616,256 rows, 10,300 local track IDs, 5,232 pedestrians.
35.85 s summed source computation, including one pilot reused by the full run.
Full repeat computes all 60 receipts exactly; no counts selected by performance.

At stride 12, 249,384 complete pedestrian labels include only 328 exact-start
windows / 78 local tracks / 84 disjoint spans under the fixed proxy. Stop and
turn proxies cover 798 and 1,612 pedestrian tracks. This is not directly
comparable to the older 31 ETH/Hotel stationary IDs: event definitions differ.
Disjoint is not IID, tracks may recur across videos, and eight scene folders
are not sixty independent calibration scenes. Do not oversell sample size.

98.388% generated rows; all bracketed by source ungenerated controls, but only
69.81% within 2px under nearest-control linear reconstruction. Do not present
bracketing as proven interpolation ancestry. Post-query-control counts are
potential provenance diagnostics. All access remains offline annotated.
Separate `control_span_coverage.json` avoids conflating directly sampled controls
with controls somewhere inside the observation span (pedestrian stride12:
5,802 vs112,175 histories with at least two, 1.90% vs36.79%).

57 focused tests pass, 635 real future mutation/truncation checks. Original
receipts are preserved; full legacy tests not rerun. Aggregate SVG rendered and
inspected. Plot font-cache warning used temporary cache and finished, not a
training/runtime failure. One ad hoc json.load(Path) inspection error corrected;
it did not change scientific outputs.

Pilot55003, full40042/PID17718, exact repeat33687, coverage/plot51106 and
tests13945 all observed terminal exit0. No active process or new CREATE job.
Public `sdd_state_support/` contains aggregate reports, hashes and original SVG;
raw data and per-video cache stay ignored. Do not touch the 3,019 unrelated
staged entries (fingerprint unchanged).

Config `configs/m3w_sdd_state_support_audit.json` digest
`9339593706eeec61c2e0affbc64bc593c1875c1d1d80953392def071a90bec0c`.
Commands `scripts/audit_m3w_sdd_state_support.py` and `--verify`, followed by
`scripts/analyze_m3w_sdd_state_support.py`. Source-control-span analysis rereads
source labels without fitting anything. No new sampling choice or role admitted.

Next substantive dependency: pending user auxiliary-source admission/sampling
decision, not another threshold scan. Do not repeat the unanswered question,
silently pool SDD, replace primary, open sealed roles or call this contribution
evidence. Current 8-to-12 / past-normalized ADE / ETH-Hotel-grouped-Zara unchanged.
Any admitted source experiment must compare matched event/track sampling and
report scarce exact starts separately. Stage5C and SMC remain disabled.

## SDD Past-Image Reader Verified (2026-09-17)

This turn is progress: new reader, fixed-prefix real extraction, independent
decode/replay and future-row mutation checks. Goal active, not submission-ready.
Previous repair pushed as `1399bdc8`. No new forecasting model or source role.

Code: `m3w_sdd_past_images.py`; builder and independent verifier share
`scripts/build_m3w_sdd_past_images.py` (`--verify` selects replay). Config
`configs/m3w_sdd_past_image_audit.json` SHA256
`12f259eac78a091c4423c3381c3df6aa91155ac286748abb7c568a7cd9f972bc`.
Data under ignored `data/stage_cvpr2027_experiments/sdd_past_images`;
public evidence `outputs/publication_readiness_2026_09/sdd_past_images`.

All 60 first-64-frame prefixes complete: 3,840 frames, 79,680 annotation rows,
44,332 non-lost, 35,348 lost, 4,231 partial crops, 302 suspect-border crops.
Build sums 81.67 s and 659,413,440 array bytes. No rows dropped for missing
history. Two Nexus prefixes have no non-lost tracks. A query population requires
a non-lost observation at/before query, not eventual track appearance.

Both original observed RGB and border-masked RGB retained. Near-black threshold
8 with current-image four-connected edge components is inferred-only; dark
boundary objects may be flagged, interior black retained. Occlusion is a source
flag, not a body segmentation. Offline interpolation provenance remains a limit.

54 focused tests pass. Independent 180-frame decode yields 2,074 bit-exact crops,
172 eligible future mutation checks pass. Completed resume verifies 60 receipts,
180 source hashes and unchanged 121 metadata/receipt/public-report files, with
no new decode/array writes. Four private sheets (12 displayed frames) self-audited,
not human gold. Full legacy suite not rerun. No model contribution established.

Pilot session72045, full76957/PID15477, replay62249 and resume-check23682
are all observed terminal exit0. No active process from this diagnostic remains.
No new CREATE query/job; unchanged access condition was not re-polled.

Next: source-role and sampling registration before auxiliary training. Pending
user question remains unanswered; do not repeat or silently admit SDD. Current
ETH/Hotel/grouped-Zara fit, primary past-normalized ADE, 8-to-12 task, sealed
Students/development/calibration/confirmation and raw t+50 supplement unchanged.
The first-64-frame check is not a representative forecasting cohort or a new
independent test. Do not confuse masks with dynamics lift or previous failures.
Stage5C and SMC disabled. Use explicit Git paths, preserving unrelated staged
fingerprint `c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

## SDD Full Media Audit and Diagnostic Repair (2026-09-17)

Previous turn was progress:36newnegative neuralfits/replay/docs pushedc705a33c.
This turn also changes authoritative state:full60video decode, new image-coordinate
helper and fixed Nexus media-link repair, rather than more loss/gate searches.
Goal active, not submission-ready. No new model trained. Newsourceadmission
question remains unanswered; do not repeat it or silently add SDD to fit.

Full audit `scripts/audit_m3w_sdd_media_alignment.py`:522,497decodedframes,
10,616,256annotationrows,346.84s summed per-video time.60header counts match,
no PTS anomalies,60annotationhashes match localOpenTrajmirror.54ref/videodimension
mismatches; same-name framecoverage only57/60. Nexus2/6/7 mismatch causes300,646
uncoveredrows(198,128visible). Alloriginalreceipts preserved; no raw edits.

`m3w_sdd_image_coordinates.py`: reference-box-edge resizeonly, statecoordinates
unchanged,past/current frameguard,noimplicitclipping. Structuraloutsidecount
3,217,812->148, maxroundtrip2.28e-13. Notpersonidentityorvisibilitycertificate.
`m3w_sdd_source_links.py`: fixedlexicographic reorder explainsNexus10changedlinks:
annotation2->media4,3->5,4->6,5->7,6->8,7->9,8->10,9->11,10->2,11->3;0/1same.
All12Nexus coveragepass;10changed medianfirst-referencecorr.1344->.9743.
Actualcompressionhistory notinspected, keep inferencewording.

Public evidence `sdd_media_alignment/`:audit.json,resize_mapping.json,
nexus_identity.json,diagnostic_media_links.json,verification.json,
link_verification.json,conclusions.md,visual_review.md. Allsource/image-derived
figures remain private data; no publishedrawmedia. Diagnosticlinks explicitly
disallow training/evaladmission, preserveannotationskeys/splits andrawpaths.
35focusedtests pass;fulllegacysuitenotrerun.6five-framecontactsheetsselfaudited,
notgold; rawJSONnot_revieweddescribesgenerationtime, reviewnoteis separate.

Pilot30691/PID11233, full9878/PID11282, cachedverify83960/PID12218,
resize71610,Nexus40974,manifest57885allterminal0. No active decode/train.
Full completedresume verified60hashreceipts,0newdecodes. Resize84sampleframes
exact,Nexus60samples exact (overlap,not144independentframes). NoOpenCV/Torch in
decoderprocess. Localfree disk77GiB duringrun; CREATE notrequeried/nonewjob.

Next independent safe action:diagnose and implement past-only partial/black-padded
image support under diagnosticrole. RawNexus3->media5frame0hasmassivewarpedblack
border; boundsmask alone would treat missing image as observed. Do not erase rows,
pretendsemanticallframespass, resampletime silently or registertraining without
source-role decision. Existingoffline annotation mode still required (98.388%
SDDrowsgenerated). OpenTrajestimated_scales.yaml exists butREADME sayssomeguesses;
notverifiedmetric. This discovery doesNOTexplain ETH/UCYnegativefits.

CurrentfitETH/Hotel/groupedZara, primarypastnormalizedADE,8obs12future remain.
NoStudentsdevelopment/calibration/confirmation, Stage5C/SMC, physicalseconds/meters
or deployment. OldStage26/37scoresexploratory. Unrelatedstagedhash remains
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`;
explicit-pathcommitonly, do nottouch3,019otherstagedentries.

## Native Spatial Motion Complete, Negative (2026-09-17)

This is a concrete progress turn, not a repeated status/blocked turn: registered
source repair, full native-crop extraction,36real neural fits,144,000updates,
independent replay/diagnosis and a new metadata-only source inventory. Overall
goal remains active and not submission-ready. Latest conclusions:
`spatial_motion/conclusions.md`. Registration committed `b0a540ca` before outcomes,
config `configs/m3w_spatial_motion.json`, SHA
`5ce0947c84a02748cf8fa3d6f9cdaca07c054a5dc4b91c5e45a69df711875414`.

4arms quality_control/lowpass_pool/lowpass_grid/native_grid; row_log only,
3seeds17/29/43,3physicalfitfolds,4,000updates.11,966rows unchanged. Native96crops
vs32blockmeans upsampled96 use identical fixed flow estimator; matching1,260D.
Common quality contains both views' consistency, not pure geometry. Crop build
133.13s; all30,013source downsamplings exact;16,106pairs;1,177,292,552localarraybytes.
Train174.57s small cached-featureMLP. No full M3W/end-to-end video training claim.
GainsvsCV-1.0260/-1.0365/-1.0810/-1.1385%;0/36positiveheld,0/36easy. Native-grid
vslowpassgrid-0.05693%,exploratory3sceneCI[-0.99781,0.07018]. Traininggainmax2.236%.
Binaryfixedcandidate/CVoracle.325--.345%only. Allnativeperrecordingseedmeansnegative.

StationarynativefeatureclippingETH100%,Hotel90.14%, but this any-column support
diagnostic is not proof clipping causes all failure. CurrentrowlogHotelstatic
positiveharmshare22.20--61.66%, NOT previousADEarm99.9%; do not conflate them.
Same31staticIDs/365overlappingrows. Fullresolutiondidnotaddindependentsupport.
All36checkpoints/predictionsreplayexact; completedresume0updates andbyte-identical
weights/report. Native32sampledcropsredecoded exactly in separate no-OpenCVprocess,
64flowpairsexact; initialbuildduplicateAVFoundationwarning retained.17focusedtests
pass;fulllegacysuitenotrerun. Plotaggregateonly, rendered and inspected.

Build5277/PID8661,pilot39255/PID9024,training59874/PID9098,replay91280/PID9603,
verification39598,independentdecode/flow66689,metadata64401,plot28958allterminal0.
No live training/process from this study. Do not launch duplicate resume.

Pending async question, do not repeat: admit local SDD as separately registered
auxiliary training/pretraining, keeping current protocol/roles unchanged? No
answer received at handoff. Metadata-only `sdd_media_inventory/headers.json`:
60readablefiles/8folders,1,195,953,035bytes,allheaders2997/100butnotverifiedtime;
9filesunder1,000headerframes. No annotations/fulldecode/trainingadmissionyet.
New source admission must be explicit; header availability is not complete or
aligned source proof. Keep old SDD exposures, scores exploratory, no metric/time.
Continue independent source-integrity/inference-path work; do not invent approval
or wait idle while safe work exists. Do not silently replace primarypastnormalized
ADE or openStudentsdevelopment/calibration/confirmation. NoStage5C/SMC.

Unrelated staged inventory still hash
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.
Use explicit-path commits only; do not disturb 3,019 unrelated staged entries.
CREATE prior SSH access blocker unchanged, not queried/no new remote job.

## Observed-Motion Study Complete, Negative (2026-09-17)

This goal turn made concrete progress: new causal-input implementation,54real
matched neural fits,216,000updates,diagnosis,exact replay and reports. Not a
status-only/blocked turn. Overall goal remains active, not submission-ready.

Latest `observed_motion_v2/conclusions.md`. Fixed OpenCV4.13.0.92 arm64 headless
isolated underignored `data/stage_cvpr2027_experiments/optical_flow_runtime`.
16,106 unique past image pairs; crop96/32 and integer-center correction followed
by supplied row/column H. No physical-motion claim. Full11,966fit rows retained,
Zara03missing images masked. 2independent builds(7.93/7.92s)matcharrayhashes.

Register85edc742beforemodeloutcomes; v1runtimeheartbeat duplicatepid atpilot100
andresume400, noheldeval. v2adb333f9preservesfailedsource/checkpoint andchanges
loggingonly. `configs/m3w_observed_motion_v2.json`, train/eval script
`scripts/run_m3w_observed_motion_v2.py`. v2trainedall54,196.69sfit, CPU4/interop1,
workers0. Input build73244, pilot38872, training5198/PID5576,replay87637,
diagnosis69733,plot24402,verification69199allterminal0. No live study process.

Quality/magnitude/direction rowloggains-0.996/-0.843/-0.972%; ADE+harm
-226.739/-201.360/-188.996%.0/54heldpositive,0/54easy. Traininggainmax52.59%.
DirectionvsADEcontrol+11.55%butexploratory3sceneCI[-21.23,15.64],notCVgain.
Nativeperrecordingseedmeansallnegative. Hotelstationary99.75--99.95%positive
ADEharm. ETH/Hotelstationarymotionfeatureclipping88.89/71.83%,only5/26IDs.
Motionmagnitude departureAUC.674/.586descriptiveoverlappingrows,notnewclassifier.
54predictionsbitexact; completedresume55weights/reportfilesunchanged,0updates.
13focusedtests pass; fulllegacy suite notrerun. PlotSVGaggregateonlyQAchecked.

Next: actual person-motion/state observation and independent start/stop support.
Do notrepeatloss/gatinggrids or callconsistencyconfidence. Before anotherrepair,
separate full-resolution actor/background/crop motion and inspect available
authorized fitting assets for independent statechanges. Broader sourceadmission
or primarychange requires explicitprotocoldecision, notquiet test reuse.
NoStudentsdevelopment/calibration/confirmationaccess, Stage5C/SMC, deployment
or metric/seconds claim. PriorhistoricalStage26/37scoresremainexploratory.

## Objective-Alignment Study Complete, Loss-Only Repair Failed (2026-09-17)

Previous turn is progress:36realvisualfits/diagnostics/negativeevidence, pushed
through28cb70f1. Newhypothesis isolates logloss and row-vs-scene sampling mismatch.
Samegeometrynetwork, same11,966fitrows, fivearms,3seeds,3folds,4,000updates each.
Decision`objective_alignment_decision.md`, config`configs/m3w_objective_alignment.json`.
Registration SHA277c1296ed4ed1919687721f13a28a17a36839c7f202df75577bfcb577b95610.
12focusedtestspass; real100step pilot0.121s saved/noheldeval. Geometry-only fast
forward exactly matches oldnetwork; constanttrainingdimensionszeroed inallarms.
Arm64venv/Torch2.12.0CPU4/interop1/workers0verified. No role/primary change.
Full45fitscomplete,180,000updates,174.57srecordedfit; trainingPID278/session2022
exited0. Replay84616,diagnosis70239,resume1280,plot95387 allterminal0; no live
study process. Do not restart. Completed resume preserves45weights/reportexact;
45savedpredictionsbitwisereplayed. Reportsunder`objective_alignment/`.

Gains rowlog-1.03,rowADE-165.67,scenelog-1.47,sceneADE-234.53,+harm-237.24%.
Everyheld/easygatefails. Trainingprimarygains upto46.40%; no fittingfailureclaim.
Hotel99.85--99.95%ofpositiveharmfromstationaryhistory underADEarms. ETH static
past/future22/81 vsHotel155/284,descriptiveoverlappinglabelsnotindependentpeople.
Allnative-coordinatearm/recordingseedmeansalsonegative; no primarymetricchange.
BinaryCV/candidateoracle.32--1.22%only; cannotrescuewiththresholds. 2kpairedscene
bootstrapexploratory3sites,notindependentconfirmation. Resultsregisteredcd197bbf;
currentrunnotmedium/fullM3Wclaim. Goalactive,no deployment/Stage5C/SMC.

Nextmostvaluable: better observed state/direction and independent start/stop
support. Do not rerun generic contextscale conditioning (already testedv6),
unboundedloss amplification or gate grids. Review past-image/motion correspondence
and input representation, then register a falsifiable candidate-quality repair.
External pretrained features require provenance/exposure review first. Do not
open sealed roles or change primary because these results are negative.

## Full Fit-Cohort Offline Visual Experiment Complete, Negative (2026-09-17)

Observation decision resolved by user delegation: standard offline annotations,
not strict sensor-as-of. Existing bound protocols/reports remain immutable.
Primary still past-normalized 8-to-12 ADE. New decision/registration:
`offline_visual_forecast_decision.md`, `configs/m3w_offline_visual_forecast.json`.
Previous turn was verified input-repair progress; this turn completed full inputs,
36 actual neural fits, paired analysis and failure diagnosis, not an audit-only loop.

11,966 complete fit windows, 3 physical-scene folds, 3 seeds, 4 arms including
mask-only control. Zara videos grouped; Zara03 kept with zero imagery. Inputs
157,241,442 bytes. CPU4/interop1/workers0. Fixed2,000 updates/final fit,
72,000 total,32.48min summed fit. Training PID92582/session98981 exited0;
all diagnostic/replay sessions also exited. No live process from this experiment.
Checkpoint+heartbeat under `data/stage_cvpr2027_experiments/offline_visual_forecast`.

Main gains vsCV: geometry-.5767%,mask-.5692%,currentRGB-.6627%,pastRGB-.7401%.
All36held/easyfailures; CV was also training-selected strongest everyfold.
36checkpoint predictions exact. Completed resume changes none of36weights/report.
107focusedtests+2supporttests pass; not full legacy suite. 2,000pairedscene
resamples with only3historicalfit sites are exploratory, not independentCI proof.
Students development/calibration/confirmation closed. No submission/deployment claim.

Completed fixed-model support repair does not turn any result positive.
Full training loss improves3.87--14.91%, but heldfitnegative.31stationarysourceIDs
stillaccount89.37%of equal-sceneCVprimaryerror. Perfect CV/candidate binary
oracle gain is only.1887--.3086%overall, so threshold tuning cannot rescue these
fixed predictions into a substantial result. Full `offline_visual_forecast/conclusions.md`.

Next: improve directional prediction support, not another fallback sweep. Register
a compact observed-motion representation/objective repair with fixed CV/geometry
controls; exclude unsupported metadata by construction. Do not change parent
primary or open development/calibration/confirmation to choose it. Broader
independent start/stop scenes remain necessary. Core joint-intervention contribution
still unproven; historical Stage26/37 results exploratory, not certified deployment.

The pending-observation wording below is historical and superseded by this entry.

## Masked Past-Image Store Complete: No Live Job (2026-09-17)

Latest `zara_masked_images/conclusions.md`. Previous turn sourceadapter repair
was progress; this turn actual masked input implementation + full source build,
not just another status audit. All14561 source rows decoded,12098 past8 windows
retained,1935 have partial crops.1962 source crop rows partial,nonezero/undecoded.
Fixed96crop/32output, meanvalid pixels per3x3 block, coverage0..9storedseparately.
No recentering or hallucination, blackpixels distinctfrommissing. Arrays60,977,761B.

Core `src/world_model/m3w_masked_history_images.py` lazyreader explicitoffline
orcontrol-as-of diagnostic modes; formaltraining/eval rejected pendingdecision.
Source latestcontrolprovenance notinfeatures. Onlypastindices/nativexy/RGB/masks
returned, nofuturetargets. ActualstrictAPI accepts111/186 andrejects3877/7924
without silentlyfilteringcohort. Thisdoesnotcertify onlinehumanidentity/sensors.

Build script `scripts/build_m3w_zara_masked_images.py`, reg
`configs/m3w_zara_masked_images.json`, SHAa02986783d6e6f1872d23cc3427febad22e86421556f519fd8130ef9c202eea3.
Sourcesboundimmutable. Build9.74s; independentfullrebuild9.68s exact18arrayhashes
andnumericfields. Completedcacheresume0newconversions,0.80s,originalreportunchanged.
18focusedtests pass; nofulllegacysuitererun. Trainingnot_run. Session90520(build)
and3954(rebuild) bothterminalexit0. NoHPCrequest/processactive.

Verifierusesprior24fixedhistories and192pastindices; old159complete masks exactly
reproduced,all33oldmissing retain>=52.08%realpixels.6privatecheckerboardinspection
sheetsgenerated,only2pagesinspected(Zara01page2,Zara02page0),notgoldlabels.
Wholearraycacheandimagesareignoredlocaldata,lightmetrics/codeonlyGit.

Userhasnotansweredofflineannotationversusstrictsensor-as-ofdecision fromprior
turn. Earlierprospectivenative-primarymetricdecisionalsoopen. Noformalnewimage
cohortortrainingadmitted,old8/12parentprimaryunchanged. Donotrepeatquestionor
inventapproval. Nextscientificworkdepends onobservationdecision; bounded
independentrepairdonehere. Do not endlessly add audits/manifests or re-run same
successful inputbuild. Broaderfit-only predictivecomparison thenneeded,plus
independentconfirmationsupport. Goalactive/incomplete; Stage5C/SMCoff.

Below are historical snapshots, superseded by this section.

## Zara Source Repair Complete: Observation Decision Pending (2026-09-17)

Latest `zara_past_media/conclusions.md`; no live job. Fresh source lineage audit
and actual coordinate-adapter implementation, not new neural training. Original
audit `zara_media_lineage/audit.json` retained, including failed direct Zara02 H.
Zara01 all5024 rows match H.txt/image rowcol/storedframe-1. Zara02 all9537 rows
match same plus one-first-exact-control origin[-1.297826082,-15.6530874004].
Maxnativeerrors5.012e-6/5.397e-6, inversepixel .000232/.000250. NoH/lag/scale fit.

Stored agents148/204; completepast8 agents148/202, windows3988/8110, full8+12
labels2234/5741. BOTH VIDEOS ONE PHYSICAL SCENE. Neither adds stationarypast
windows. Most stored pasts use a later interpolation control:3877/3988 and
7924/8110. Only111/186 allcontrols<=query; nofilter applied, notstrictsensorproof.
This is distinct from train/test leakage and standard offline annotation tasks.

Async question sent this turn: keep standard offline annotated forecasting with
explicit non-online limitation(recommended), or require strict source-as-of
observations/new protocol? Unanswered. Earlier prospective native-primary-metric
question also pending. Do not silently change current parent primary/split/rows.
Independent work can continue on mask-aware input support, not formal cohort
admission until necessary scientific decisions arrive. Do not repeat questions.

192requests/80indexedframes,159validcrops,33missing,24first-historyIDs,6local
contact sheets inspected. Some points near feet, others head/body, occlusion;
no manual gold pose/identity proof. Zara02 first12 windows all have some missing
crop support. Complete-image rejection would bias entering agents; implement
explicit partialmask for next admitted experiment. Nofuture-survivalfilter.

10focusedtests passed. Separate actual replay matches allnumericfields and
privatecache/contact hashes. Sessions65200 exit0(original), replayexit0;
sourceauditPID69160/mediaPID69827 are historical, heartbeatscomplete. NoHPCrequest.
RegSHA43d8a468...lineage and522964e9...pastmedia; boundfiles immutable. Raw/caches
underignoreddata only. Formaltrained/confirmatory/metric/secondsresult not_run.
Goalactive,incomplete;this turnmade concreteprogress,notrepeatedhardblockturn.
Need larger moving-context fit comparison after semantics approval; retain all
negative forecasts. Stage5C/SMC off. CREATE blocker unchanged, no repeatedlogin.

Below are historical snapshots, superseded by this section.

## Camera/Support Repair Complete: No Live Job (2026-09-17)

Latest: `appearance_support_control/conclusions.md` and
`appearance_no_camera/conclusions.md`. Previous turn was progress; this turn
also produces new inference controls, real retraining and verified negative
evidence. The overall submission/research goal remains active and incomplete.

72 settings =18 frozen predictors x original/camera-box/all-box/support-fallback.
No setting positive. Every held row outside at least one marginal train feature
range, so support fallback rejects all365 rows: exactCV, not successful transfer.
Past-RGB guarded means ETH -14.98 -> -16.15%, Hotel -172.63 -> -82.86% (all-box).
Same-mask decomposition shows some apparent harm reduction is fewer switches;
geometry-only Hotel camera-box fixed-mask prediction is worse than original.

Then6 real matched refits remove camera input columns28:32 after normalization,
keeping deterministic outputH and every other setting. Seeds17/29/43,2folds,
1,000updates each,6,000total;109.87s summed fit. New guarded ETH -19.6355%, Hotel
-86.2760%, all6 negative. Original images/masks/28context features/loss/budget
retained. Easy ratio undefined atzeroCVfloor; absolute harms in reports.

Registrations `configs/m3w_appearance_support_control.json` (9e2f1ea84743...) and
`configs/m3w_appearance_no_camera.json` (59e0e101e1da...). Do not change bound
sources/decisions/reports/checkpoints. Both original18 and new6 models replay
exactly, resume0newfits/evals with18/6cached receipts.15 focused tests pass across
13+2 invocations; full legacy suite not rerun. Control session29594 exit0;
training1153 exit0 historicalPID67849. No live job or new HPC request.

Next useful work is broader verified fit-context support, not more box/threshold
sweeps. Current approved fit roster ETH/Hotel/Zara01/02/03; localZara01/02 video,
reference andH files exist. This is file availability only, not mapping/source
admission. Zara03 lacks video in inspected local directory and retains packaged
population limitation. Zara recordings are one physical scene, not independent
sites. Inspect coordinate/image mapping and causal history population before
planning broader multimodal fitting; don't reopen developmentStudents to tune.
New prospective native-ADE/FDE primary still pending; preserve current8/12parent
andraw50supplement. No new deployment, final-test claim, Stage5C/SMC or paper
readiness. CREATE access/project blocker unchanged; no repeat login needed.

Below are historical snapshots, superseded by this section.

## Past-Appearance Neural Probe Complete: No Live Job (2026-09-17)

Latest: `past_appearance_probe/conclusions.md`. 18 real CNN/fusion fits, 3 seeds,
2 held fit scenes, geometry/current-RGB/eight-past-RGB. 1,000 updates each, 18,000
total; 135.13s summed fit CPU4/interop1/workers0. Small diagnostic, not medium/full.
Registration `configs/m3w_past_appearance_probe.json` SHA
`e190e0b0f87343c356de3aabd979859223d5aee507add9f1519bb9f181b1fd5b`.
Do not edit bound code/decision/sources or overwrite old outputs.

365 stationary rows, 31 agents, 45 runs; ETH 81 rows/5 agents, Hotel 284/26.
2,920 requested patches, 458 unique indices, 2,664 valid; 333 full windows,
32 masked rows retained. Input cache has only whitelisted past metadata; targets separately
loaded for loss/eval. Source index convention remains an assumption, not clock
verification; H provides coordinate mapping, not metric calibration.

None of 18 guarded settings positive. Mean guarded ETH geometry/current/past
gains -0.0133/-8.0940/-14.9761%; Hotel -132.0355/-167.7268/-172.6322%.
Easy ratio undefined at zero CV floor, absolute harm preserved. All held Brier
lifts negative. Training on ETH fits trajectories (65.09--68.95%gain), fails
Hotel. 78.52% of Hotel rows have features beyond the ETH standard-score clamp;
Jacobian 67.61%, unseen circle/radius 40.14%. This is measured support shift,
not proof that clipping alone causes failure. Hotel train forecasts already
worse than CV.

All 18 checkpoint outputs replay exactly. Resume 0 new fits/evals, 18 cached_verified,
original completion preserved. Four focused tests passed, including exact
optimizer resume; full legacy suite not rerun. Full training session99386 exit0,
historical PID65787; pilot44728 exit0, 100 updates resumed; replay46110 exit0;
diagnosis67155 exit0. No live process/HPC job from this work. Full row predictions,
checkpoints and input cache stay ignored. Public metrics omit training row
indices/normalizers. New training is not a NumPy fallback.

Next: predeclare a camera/support treatment on frozen predictors to test the
measured mismatch, before larger training. Do not retune 0.9 or select winning
seeds from these held outcomes. Need utility/harm, not start confidence alone.
Broader independent sites remain critical; further capacity on five agents is
not a paper contribution. Pending NEW native-ADE/FDE primary decision still
unanswered; parent 8/12 primary unchanged. No independent confirmation,
deployment, Stage5C/SMC or submission readiness. Goal active, incomplete; this
turn progress. CREATE blocker unchanged, no repeated login.

Below are historical snapshots, superseded by this section.

## Moving Past-Image Controls Complete: No Live Job (2026-09-17)

Latest: `past_motion_comparison/conclusions.md`. Fresh radius24 and registered
adaptive radius64 image-correspondence diagnostics, NOT new forecast training.
Same24 fit-only moving agents (12ETH/12Hotel), first complete moving8 histories,
hash-selected without future availability/labels. 192requests/187distinctframes,
4Hotel centered crops missing. All6 original contact sheets inspected. Markers
often near head/upper body, no verified pose/body boxes. Centered crop corrects
the earlier ungrounded above-point body assumption without certifying identity.

ETH original median match errors1.41/1.00px. Hotel56/84displacements outside
24search; inside28errors2.52/1.78px. Radius64 same-support Hotel errors
22.12->7.66 and16.71->4.22; ETH worsens3.07->6.49 and3.60->7.51.
Wider missing ETH4/7,Hotel8/15 by template. Do not choose a per-scene winning
radius, treat missing as zero, claim future ADE lift or call ZNCC pose confidence.
Direct native-index plausibility improves; capture clock/identity not certified.

Official ETH CVL dataset page checked live: research-use scope with citation;
no redistribution authorization inferred. Original paper PDF direct404; don't
claim its detailed point semantics read. Both streams25, ETH helper15 unresolved
for seconds. No video timing offset fitted. Source metadata preserved unchanged.

Sessions56815 and71987 exited0; historical PIDs64007/64231 heartbeats complete.
Fresh processing~6.90/12.59s. Paired summary validates source/completion hashes and
identical controls.14 focused tests pass0.57s; full legacy suite not rerun.
No current job, no neural training or HPC request. Rawframes/crops/rows under
ignored data/stage_cvpr2027_experiments/past_motion*, aggregate reports public.

Next useful work: register a fit-only point-centered past-appearance versus
trajectory directional/forecast probe under declared native-frame mapping, with
explicit missingness and trajectory fallback. Keep old parent metric, eight-step
observation, folds and negatives; pending NEW native-ADE primary still unanswered.
No further threshold or temporal-offset tuning on exposed results. Moving sample
checks do not show stationary orientation is inferable. Need actual predictive
ablation before contribution claim. Independent sites remain a separate blocker.
No new deployment/submission success, Stage5C/SMC. Goal active; this turn progress.

Below are historical snapshots, superseded by this section.

## Past-Video Audit Complete: No Live Job (2026-09-17)

Latest: `past_video_alignment_v2/conclusions.md`. Fresh fit-only decoding, not
new model training. 31 first-stationary-query agents, first/last of eight observed
indices, 62 image requests / 48 distinct frames. All retrieved; 58 unclipped
inspection rectangles, four clipped. Six local contact sheets inspected. This
does NOT verify full body coverage, identity, capture clock or modality admission.

Upstream plotting reverses inverse-H output to image xy. Hotel inside count
5,417 -> 6,533 of 6,544; ETH 8,908 unchanged. Preserve original availability
reports and model scores. World-coordinate XML features/forecasts not modified.
Encoded videos both rate25; ETH helper assumes15. Direct native-index requests
follow playback convention, not an independently certified as-of mapping.
Fixed crop extends80above/16below; bodies often at bottom, branches/poles and
nearby people present. Do not reuse these as verified body/pose boxes or labels.

Initial session32795 failed JSON serialization after decoding; original source
snapshot and partial images retained locally. Python-int rectangle repair gets
new registration `configs/m3w_past_video_alignment_v2.json`, successful
session80612 exit0, heartbeat historicalPID62549 complete. Decode~3.28s.
Completion hashes bind aggregate report/local row records. Do not overwrite.
18 focused tests passed0.14s; unchanged legacy suite not rerun. No live local or
HPC process from this work. PyAV18.1.0 isolated under ignored data, no Torch change.

Next: verify annotation-point semantics and past frame correspondence before a
registered trajectory-versus-past-RGB directional probe. Fix crop/missingness
design only from past/source evidence, never held outcome. Source-use admission,
independent sites and prospective primary-metric decision remain unresolved.
Do not silently approve native-ADE/FDE primary or relabel old scenes untouched.
Do not repeat completed forecasters/threshold sweeps or unchanged HPC logins.
No visual training, seconds/metric claim, new deployment, Stage5C or SMC.
Goal active, incomplete; useful source-input repair is not forecasting success.

Below are historical snapshots, superseded by this section.

## Conditional ADE Decision Complete: No Live Job (2026-09-17)

Latest: `conditional_ade_probe_refined/conclusions.md`. New one-factor predictor
decision experiment, not new neural/tree training:18 frozen ExtraTrees settings,
same365 stationary rows, opposite fit scene, three seeds/feature arms. Conditional
mean replay <=7.64e-17 normalized; replace mean with weighted per-step geometric
median using only original training labels. No held labels in weights/decisions.
Parent8/12 task/primary/gate0.9 unchanged; no development/calibration/final labels.

All9 ETH medians are exact zero/CV. All9 Hotel medians negative; no guarded median
positive. Median Hotel mean gains -90.38/-21.37/-21.25% versus original
-346.70/-213.07/-252.23% (pooled/scene/scene_neighbor). Damage reduction is not
positive transfer. Still native harm positive on Hotel; ratio undefined because
CV floor zero. Conditional optimality on train does not ensure held-scene gain.

Original23/39420 numerical waypoints missed tolerance. Source/registration/results
preserved. Separate registered training-only refinement certifies22, leaves1
approximate:fold1_seed29_scene_neighbor, query172,step9, gap2.293e-9. Never write
all solves certified. Refinement maximum native change3.122e-5, no sign changes.
18 original receipt checks on resume all cached_verified, zero recompute/refits.
Do not overwrite original metrics or bound code; replay reports use new dirs.

Main process46566 exited0; tests27470 exited0:23 passed1.16s. Numerical follow-up
and18-trial resume terminal. No live job from this work. Total point-compute
about16.1seconds; do not report it as long training. Source models/weights/row
outputs local under data/stage_cvpr2027_experiments/conditional_ade_probe*.

The conditional mean hypothesis explains some drift but not useful direction.
Next needs verified past directional context/independent support, not another
point-estimator or threshold sweep on these same exposed folds. Scientific
native-ADE/FDE-primary decision still pending; don't silently change the parent
or call old scenes untouched. No new deployment or submission readiness.
Goal active and incomplete; this is meaningful experimental progress. CREATE
access/project blocker unchanged; no repeated login attempts. Stage5C/SMC off.

Below are historical snapshots, superseded by this section.

## Precision and Conditional-CV Audits Complete: No Live Job (2026-09-17)

Latest: `stationary_label_resolution/conclusions.md`. Two fresh fit-only audits,
no new training or model selection. All 365 rows/31 agents/45 runs retained.
No changed window is explained by printed precision. Inferred integer-pixel
lineage holds for all 15,452 ETH/Hotel source rows; it is not calibration proof.
Conditional +/-0.501-pixel CV LPs reject 46/59 ETH and 115/129 Hotel changed
windows, representing 98.30%/97.21% of stationary CV error. All177 unchanged
windows feasible;27 changed feasible; zero inconclusive. All37 Hotel return
windows incompatible. Do not export feasibility or future-derived slice labels
as features. No physical start, annotation-accuracy or metric/seconds claim.

Every unrestricted frozen regressor is negative on the >5-pixel seed-mean slice.
One guarded Hotel scene-neighbor tree has +0.1444% local slice mean, but still
harm and full-subset failure remain. Original 72 corrected model artifacts were
hash-verified/replayed within1e-12; no refits. Reports bind original source/model
identities. Do not overwrite `audit.json` during rerun: use new output/report dirs.

Precision run8529 and conditional feasibility run completed exit0. Focused tests
42200 terminal:24 passed in1.07s. No current local/HPC job from this work. Legacy
suite not rerun. Aggregate reports/code/configs only are public; row statuses,
caches and model files stay local. CREATE access/project blocker unchanged.

Pending user choice remains NEW prospective native-per-dataset ADE/FDE-primary
protocol; no answer at this snapshot, not approved/run. No current metric or old
exposure changed. Next useful work needs verified past directional information
and independent support, not another precision-only fix or threshold sweep.
Do not repeat these completed audits for status. No deployment/submission claim,
Stage5C or SMC. Goal remains active and incomplete; this turn made progress.

Below are historical snapshots, superseded by this section.

## Static-Scene Probe Complete: No Live Job (2026-09-17)

Latest: `stationary_scene_probe_v2/conclusions.md`. Initial72 plus repaired72
fit-only classifier/regressor fits, no new forecast network/deployment or final
labels. 365same rows/31agents/45runs, frozen parent eight/twelve-step task. Static
obstacle proxy only; image pixels, destinations/groups and videos excluded.

Initial corner logic falsely marked35Hotel frames undefined. Source snapshots
and original registration/metrics remain preserved. New v2 fixes identical shared
corners; zero undefined frames. Corrected scene-tree AUC0.8048ETH/0.5387Hotel,
mean Brier lifts+0.00617/+0.02894. All36 corrected trajectory regressors worsen
CV, and no fixed0.9gated regressor improves. Two tiny original gated positives
disappear after fixing the bug. Easy CV error zero: ratios undefined, retain
absolute harm. Native-coordinate signs agree; changing units cannot fix this.

Jobs26005 and39330 exited0; saved-model replay32278 exited0. All144models match
predictions within1e-12 (max2.22e-16).14focused tests pass. No live job from this
turn. Data/checkpoints local under `data/stage_cvpr2027_experiments/stationary_scene_probe*`.
Resume74834 also exited0:36cached_verified pairs, zero new fits, separate local
resume_check report.24registration bindings checked unchanged. Console72model
count is inventory, not new training; no cached result is relabeled fresh.
Do not overwrite original metrics bound by the v2registration on resume; use
separate report dirs. Code/model/cache identity checks must remain intact.

Next scientific choice: an async question asks whether to create a NEW prospective
native-per-dataset ADE/FDE-primary protocol, with past-normalized ADE supplementary.
It has NOT been approved/run as of this handoff; preserve current primary results
and historical exposure. This is motivated by benchmark interpretability, not a
license to relabel a failed study. If answered, follow the actual latest answer.
Old scenes cannot become independent tests. Continue independent data/context
validity work while needed approval is pending, without requerying unchanged
CREATE access. Current signal does not justify a larger stationary-start network.

Core goal remains active, not achieved. Scene-body orientation, temporal/source
validity, predictive candidate quality, independent sites and joint-mechanism lift
remain gaps. No Stage5C/SMC, metric/seconds or submission-readiness claim.

Below are historical snapshots, superseded by this section.

## Stationary-Start Diagnostic Complete: No Live Job (2026-09-17)

Latest work is `stationary_start_probe/conclusions.md`. Only frozen fit data
were used; no new development/calibration/confirmation labels. 365 stationary
windows, 31 agents, 45 runs at ETH/Hotel; Zara held stationary comparison not_run.
All canonical source positions match cache exactly. This is not annotation/clock
or physical-stillness verification. ETH time conflict remains unresolved.

First extraction exited before fitting on NumPy-bool JSON serialization. Keep
the original registration and local failed_source_snapshot; fixed v2 registration
ran all 24 ordered classifiers. Every one worsened window Brier versus train-only
prior. A single-factor adaptive pooling repair ran 24 more. Pooled geometry trees
Hotel->ETH improve Brier +0.02513/AUC0.6949, including run/agent balancing;
ETH->Hotel remains negative -0.01943/AUC0.4895. Positive held support only5agents.
No bidirectional gain, new trajectory result or deployment. No stationary-start
neural residual was launched. Do not interpret Brier differences as ADE gains.

Jobs 25466/83206 and replay51992 all exited0. Forty-eight checkpoints replay Brier
within1e-12; seven new focused tests passed. No local/HPC task from this study is
still running. Row caches/models remain under ignored data/stage_cvpr2027_experiments.
Code+configs+aggregate reports are the only new public artifacts.

IMPORTANT: pooled registration binds the ORIGINAL `metrics.json`. Do not overwrite
it with resume reports whose statuses change to cached_verified. Use the runbook's
separate replay report path. Neither original failure registration nor its source
hashes should be rewritten; v2 is the successful source/version identity.

Next: fit-only scene-cue availability and relative start-direction identifiability
audit without extending the eight-step observation, dropping stationary cases,
or changing the primary metric. Check eligible image/context alignment before
another start-aware forecast; no unsupported ETH timestamp mapping. Two-site
support and direction prediction remain gaps. Do not requery unchanged CREATE
access or re-run completed predictors just for status. Goal remains active,
not achieved and not at a hard impasse; no Stage5C/SMC.

Below are historical progress snapshots, superseded by this section.

## v7 Real Deferral Complete: No Live Job (2026-09-17)

This goal turn made substantive progress: registered and completed the missing
real cost-sensitive deferral control on both frozen v7 families, all three seeds,
four fixed head/cap settings. 24 heads x1,000 updates, 11,966 identical OOF rows,
306 features. No predictor retrained and no old protocol/source bytes changed.
The first real fit smoke job and full resumed exec37123 both exited successfully;
PID53207 is historical. Completion and heartbeat under
`data/stage_cvpr2027_experiments/8to12_deferral_v7` establish terminal status.

Code/registration launch snapshot `eca752be` is pushed. Summary and conclusions:
`8to12_deferral_v7/{metrics.json,metrics.csv,results.md,conclusions.md}`.
All 12 fresh recording evaluations exactly match parent forecast errors before
hash-verified ordinary decisions are reused. 480 paired comparisons are not
independent runs. Deferral routing is unconstrained, not coverage/risk matched.

Skip gains all negative. Bounded gains +0.01997--0.08445%, but easy degradation
14.40--96.96%; all 24 fail positive gain plus easy<=2%. No new deployment,
objective-superiority or paper-readiness claim. 42 focused regression checks and
eight table checks pass. Full legacy tests were not repeated. One exposed site
still prevents independent scene CI. Loss decreases do not establish convergence.

Next useful experiment: fit-only stationary-to-moving identifiability, separating
annotation/normalization artifacts from real starts and testing past neighbor or
scene cues against a fit-only start prior with whole-scene crossfit. Only then
register a start-aware forecaster if warranted. Do not retune the same routing
thresholds, alter the primary metric after inspecting outcomes, or disguise old
development as confirmation. Independent sites and source/time validity remain
missing. Do not repeat unchanged CREATE login probes. Goal remains active.

## v7 Complete: No Running Experiment (2026-09-17)

Both registered families finished all three seeds, 24 forecasters, six neural
cost heads, six ridge controls, primary evaluation and both frozen supplements.
Main exec21687 and supplementary exec83736/89872 have exited successfully. All
summary/replay/diagnostic jobs are terminal too. Do not restart training just to
refresh status. Periodic supplementary heartbeats may remain running; verified
completion receipts and terminal process results establish completion.

`8to12_residual_pair_v7/conclusions.md` and `supplement_conclusions.md` hold the
current evidence. CV-skip mean uncontrolled gain -0.59857%; bounded +0.06151%.
Bounded uncontrolled easy degradation 62.58--398.57%, so it is not deployable.
Selected development gains are only +0.002495%, +0.007710%, +0.000519% with easy
1.189%, 1.883%, 0.251%. All selected bounded arms are independent, not joint.
Candidate/CV oracle headroom <=0.33702% rules out a 5% gain through further
routing of these same predictions under the unchanged primary metric.

All fixed primary forecast errors/ordinary decisions replay exactly in both
supplements. CV-skip count control has zero identity differences. Bounded has
196 differing repeated exports; nine zero comparisons and three moderate-ridge
ADE changes, all positive (joint worse). Raw50 uncontrolled bounded gains are
+0.04217%, -0.04695%, +0.00559%. No stable joint benefit or new deployment.
24 final targeted comparison/supplement/error/headroom/replay checks pass.
The earlier verified model/resume suite remains valid; no full legacy suite rerun.

The goal remains active, not blocked or complete. This turn produced real paired
training/evaluation rather than another plan. Next work must address candidate
quality and the absence of useful joint-choice disagreement prospectively, not
retune current thresholds or change the primary metric after seeing results.
Consider the stationary-start capacity tradeoff and source/normalization audit
before another architecture. Independent source-eligible sites, compatible
benchmark geometry/time and a real deferral comparison remain missing. No
source-use roles are silently approved, and no repeated unchanged CREATE probe
is needed. Training code is preserved by `a26d2996`; fit diagnostic by `56d037bd`.
Below are historical launch/continuation snapshots, not live state.

## Historical v7 Paired Training Launch (2026-09-17)

The previous goal turn completed v6 and its supplements, giving actionable
negative evidence. This turn tests the frozen output-parameterization hypothesis
in `residual_parameterization_v7_decision.md`: exact-CV-initialized residual and
the same model with a past-motion amplitude bound. No row, label, metric, loss,
fold, seed, budget or policy threshold changes between these two arms.

Protocol `configs/m3w_8to12_residual_parameterization_v7.json`, digest
`f82ec96eaaea9ecd7ab7218829f99f43e4c91ab1c7af3ef27d38f08fd0f1621a`.
Active runner: `.venv-pytorch/bin/python scripts/run_m3w_residual_parameterization_pair.py`.
Read the actual child and heartbeat under `data/stage_cvpr2027_experiments/8to12_residual_skip_v7`
or `8to12_motion_bounded_v7`; never restart a live child from a stale periodic snapshot.
CPU4/interop1/workers0. Both real 100-step pilots passed and were copied under
`residual_parameterization_v7/pilots` before resuming their full fits.

All seeds 17/29/43, four predictors per seed at 10,000 updates, both OOF heads
and development evaluation are required. Tests: 81 core checks, 22 gain/harm
checks plus one optional-device skip, seven comparison checks, thirteen
supplement/replay checks. These are engineering results, not a predictive gain.
Old v6 is reproduced from `052bcc64`, not by changing its saved identity.

Fit-only bound diagnostic is complete: `8to12_residual_pair_v7/fit_bound_headroom.json`.
11,966 rows, 365 zero-budget rows, 73.253% of pooled fit CV error on those rows;
optimistic correction-ball headroom 4.089%. This is label-aware geometry, not
learned/development evidence or the equal-scene primary aggregation. Two tests
pass. Do not change the running bound or omit stopped-to-moving targets.

When both family summaries complete, run `compare_m3w_residual_parameterizations.py`
with `--skip`, `--bounded`, cached `--reference-v6` Transformer metrics and a new
report directory. The v7 fixed supplement decision is already written before
development: pass `--decision outputs/publication_readiness_2026_09/forecast_supplement_v7_decision.md`
to the supplementary evaluator with each completed family, CPU4. Preserve all
negative seeds, absolute easy error and missing-label denominators. Do not
retune the bound from these results or relabel development as independent test.

## Current State: v6 Primary and Supplements Complete (2026-09-17)

Both v6 families completed all registered seeds and fits: 24 forecasters at
10,000 updates, six neural cost heads at 1,000 updates and six ridge controls.
All six development choices are CV. Transformer primary gains are -5.736%,
-7.686%, -6.700%; EqMotion-K1 gains are -14.119%, -8.473%, -14.081%.
The complete pair is in `8to12_public_predictors_v6/paired_predictor_comparison.md`.
Do not restart the completed training or repeat its runtime pilot.

New error decompositions in each family's `error_scale.json` locate most positive
harm at the numerical past-scale floor, without removing rows or changing the
metric. Native Students03 EqMotion results beat the development-best causal
alternative by 0.94--2.32%, but Students01 and the primary result remain negative.
See `8to12_public_predictors_v6/conclusions.md` for the next falsifiable direction.

Both supplementary families completed all three seeds and twelve candidates.
Transformer has identical matched-count switch identities and zero ADE/FDE
difference. EqMotion changes only 76 repeated agent-query decisions, with tiny,
oppositely signed effects in seed29's two ridge policies and ten zero results.
Raw50 joint ADE gains are all negative. Both row replay audits match every
original fixed-forecast error and ordinary decision. No study process remains
active at the completion check. The last periodic heartbeat says running;
verified `completion.json` and the exited PID establish final completion.

Summary and replay reports are under each `8to12_<family>_v6_supplement` directory
in the publication output tree; large caches stay under ignored data. There is
no pending supplementary rerun. Next work must be a new prospective hypothesis,
not another status poll or threshold sweep. Main protocol digest remains
`53be3aafbda47ddf8d60891e779896f6685fcd59222a1fe4ea867e689f8914de`.
No independent-scene CI, deployment or submission-readiness claim. Stage5C/SMC
remain disabled; the research goal is active. The earlier snapshots below are
historical only, including resolved protocol-choice blockers and old run states.

## Historical v6 Launch Snapshot

The v5 seed29 full fit failed with nonfinite loss on both MPS and CPU, after
seed17 completed its four predictors, OOF heads and development evaluation.
v5 is incomplete, not a three-seed result; preserve its artifacts and the
`73b30e6a` source snapshot. Do not alter old protocol hashes to run new code.

Current runner: `scripts/run_m3w_conditioned_predictor_pair.py`, arm64 environment.
Protocol `configs/m3w_8to12_conditioned_context_v6.json`, digest
`53be3aafbda47ddf8d60891e779896f6685fcd59222a1fe4ea867e689f8914de`.
The only model factor is past-only common input conditioning, with predictions
restored before the unchanged loss/cost/evaluation scale. Decisions, source
caches, rows, seeds and 10,000-update budgets stay fixed. See the frozen v6
decision and pilot report. A real seed29 MPS 100-step pilot passed (13.01s),
and is reused/resumed when the runner reaches that seed; it is not a full fit.
48 tests passed; one optional MPS test skipped, separate real MPS run completed.

Active data directories: `8to12_eqmotion_v6` then `8to12_transformer_v6` under
`data/stage_cvpr2027_experiments`. Read their runner heartbeat and actual child
before resuming. Full/fold fits, OOF ridge, 1,000-update neural cost and development
evaluation run sequentially for seeds17/29/43. No bound-source edits during this
run. The summary and comparison helpers retain failed/negative seeds, all easy
metrics and actual runtimes. No v6 accuracy result yet; don't stop for slowness.

After both summaries, compare v6 metrics with `scripts/compare_m3w_public_predictors.py`.
One development site is not enough for a scene CI. EqMotion K=1 is not published
best-of-20. No independent confirmation, deployment, Stage5C or SMC claim.

## Historical v5 Run

Latest interruption: v5 EqMotion seed17 full/hold0/hold1 each completed 10,000.
hold2 failed at MPS step89 with float32 nonfinite output; outer pair runner exited.
The outer runner initially stopped. The explicit direct CPU resume of
seed17_hold2 subsequently completed all 10,000 updates in 2,093.36 seconds.
The original pair runner has now resumed, reusing all four completed predictors
and continuing OOF extraction. Its refreshed root heartbeat identifies the
current child. See
`8to12_public_predictors_v5/nonfinite_fit_diagnosis.md`. The MPS pre-failure
weights/batch are preserved locally. This is not the old OpenMP hang or a claim
of a root-cause fix. Do not skip folds; completed fits are hash-verified, not
retrained. Check actual current heartbeat/child before any additional restart.

Current source snapshot: `5e0f7be9`. Active command:
`scripts/run_m3w_continuous_predictor_pair.py` in arm64 `.venv-pytorch`.
Read `data/stage_cvpr2027_experiments/8to12_eqmotion_v5/runner_heartbeat.json`
and its indicated child log/checkpoint before any restart. It then runs the
matched Transformer at `8to12_transformer_v5`. Both register seeds 17/29/43,
full + three physical-fold predictors with 10,000 updates each, then OOF ridge
and 1,000-update neural gain/harm heads. Do not edit bound code while running.

v3 is an incomplete 200-update compute pilot. v4 completed seed17 full EqMotion
at 10,000 updates (1,151.87 s cumulative), then was deliberately halted before
development scoring after a source-context issue was established. Neither is
a completed three-seed comparison or accuracy result. v4 checkpoint preserved:
`data/stage_cvpr2027_experiments/8to12_eqmotion_v4/seed17_full/latest.pt`,
SHA256 `8525f739cb082a5475d8acbc933ae07abd5063bb4f397fe3d799954a6021f88d`.

The [Students01 row audit](students01_packaging_audit/audit.md) establishes exact
20-point prefix truncation and identity fragmentation, not a clock reset. All
17,820 packaged rows map to original timestamps/rounded coordinates; 3,993
short/tail rows and all 63 short tracks are absent. Their retention depended on
later availability, so past-only reader checks alone were insufficient for the
desired full-scene causal observation population. v5 uses the continuous local
source, keeping 415 original identities and 14,295 complete 8-to-12 windows.
Fit recordings, scientific choices and budgets remain unchanged. Historical
exposure is not cleared; annotation-generation causality is still unverified.

Protocol: `configs/m3w_8to12_continuous_context_v5.json`, digest
`24c0fb195ef76430f5d73ff06f1736afdef213d507c4ebd1d9cccdeba86d28e9`.
New fits restart rather than relabel old checkpoint identities. The completed
v4/v5 seed17 full fits were compared: all parameters, all 10,000 losses and
sampler order/cursor/RNG match exactly. See
`8to12_public_predictors_v5/unchanged_fit_replay.json`. This is one same-hardware
replay, not accuracy evidence; protocol identities/development populations differ.

EqMotion is the pinned public core, fixed K=1, not published best-of-20.
Unused-head execution was pruned with exact output/trainable-gradient matches
in CPU/MPS tests, without reducing selected-head model capacity or budget.
Runtime fallback cannot be silent. Current sequence is healthy local MPS/CPU,
not a new CREATE job; remote access conditions have not changed.

After both v5 summaries complete, run `scripts/compare_m3w_public_predictors.py`
with the two v5 metrics paths; retain every negative seed, easy error and native
strong-causal comparison. One University development scene cannot support a
scene CI. Source packaging changed, so v2-to-v5 aggregate changes are not a pure
model ablation. Update the run status/README/state with actual outputs, commit
only scoped light artifacts; unrelated staged fingerprint remains
`c055a883338b2eaf7f54d5bf1ce29c846df56ea4bf3e66e529bbdddc3b90c323`.

Goal still active: no submission readiness, no new deployment, no Stage5C/SMC.
Do not stop a healthy run because it is slow or replace it with status-only work.

Superseded for development on 2026-09-16: the user selected obs8/pred12 with
raw-frame t+50 supplemental and delegated the remaining research route. See
[the new decision](research_route_decision.md) and
`configs/m3w_8to12_development_v1.json`. Independent source/confirmation
requirements remain open. The earlier blocker is preserved below as history,
not a reason to ask the answered horizon question again.

## Current Executed State

Two real three-seed studies are complete: v1 coordinate MSE and v2 Smooth-L1.
Thirty neural fits completed 1,000 updates each; no active study child remains
after the runner completed all requested seeds. Checkpoints and per-fit heartbeats
are in the ignored `data/stage_cvpr2027_experiments/8to12_v1` and
`8to12_robust_v2` directories. Current main task is native obs8/pred12; raw t+50
supplement remains separate and not run.

Both studies select the CV floor in every seed. Mean uncontrolled primary gain
improves from -7.414% to -0.247%, but easy protection and joint-selection
contribution still fail. The robust candidate/floor oracle upper bound is only
0.231--0.351%; further threshold search cannot produce a 5% primary gain with
these fixed candidates. See [paired results](8to12_robust_v2/robust_loss_comparison.md)
and [failure conclusions](8to12_robust_v2/conclusions.md).

Next substantive work: investigate causal scale handling and a genuinely useful
public predictor, then run real deferral and realized-count-matched controls.
Do not build more status-only modules or repeat the same runtime probe. Preserve
all negative seeds. Native-coordinate improvements over CV alone do not beat
the stronger damped baseline consistently. Independent confirmation is still
missing, and no submission-ready or deployment claim is established.

MSE source snapshot: `707d4017`; robust source snapshot: `9e592089`.
Do not edit old protocol hashes to load changed implementations. Current code
runs v2 using the explicit command in the [runbook](local_create_runbook_zh.md).
CREATE access has not changed; no new remote job or absent-job claim. Stage5C and
SMC stay disabled. The research goal remains active, not completed.

## Historical Blocker Snapshot

Checked 2026-09-16. This is an execution blocker record, not research progress, a new gate or a completed experiment. The full CVPR research objective remains unfinished.

## Fresh Checks

- Current research commit before this handoff: `15cf26dd7cc71ffbd993f68fbb5dc00937f506cc`.
- Registered scientific protocol: `configs/m3w_independent_experiment.draft.json`; status `draft`, approval `null`.
- Protocol file SHA256: `1e6f401a5f75b22d031d6c1a2be80c3faece4ef16019d8a9778bf1cd6d2989d8`; unchanged.
- Nine recordings, six declared physical scenes; all nine retain `development_exposed` and `unassigned` roles.
- History length, prediction unit, horizon, primary metric and aggregation remain unspecified. Calibration risks, folds and seed registration are also incomplete.
- `.venv-pytorch/bin/python scripts/train_m3w_neural_cost_head.py --preflight-only` returned exit 2 with `Explicit protocol approval required` and `neural_cost_training_started: false`.
- A read-only host process check found no matching local M3W/stage/WorldCore/Pytest training command or CREATE SSH/sbatch command at inspection time. This is not a remote scheduler inspection or proof about unrelated processes. The first sandbox process check was denied; the authorized read-only check succeeded.
- The old draft binds earlier versions of `m3w_experiment_contract.py` and `m3w_joint_intervention.py`. Their hashes now differ after the documented repairs. These bindings must be reviewed and refreshed when a new protocol is frozen, not bypassed by changing only `status`.

## Existing Evidence, Not Recomputed Here

The [neural cost-head report](neural_cost_head/implementation_and_limits.md) records synthetic CPU/MPS fitting, recovery and comparison checks. The [support audit](risk_calibration/support_audit.md) records the lack of approved independent calibration scenes. Neither supplies a clean real forecasting result.

CREATE access previously failed with public-key authentication and a portal MFA notice; the project path is unresolved. No changed access condition was supplied, so the same failed connection was not retried. Remote jobs and artifacts remain unknown, not absent. No new job was submitted.

## Decisions Still Needed

Two previously raised scientific choices remain unanswered:

1. Use an 8-observation/12-prediction benchmark task as the main comparison, with raw-frame t+50 supplemental, or retain raw-frame t+50 as primary? Observation steps must not silently become seconds. The final approval must also specify primary ADE/FDE, aggregation, data roles and folds.
2. Prioritize an explicitly empirical, development-exposed matched-control study while seeking independent confirmation, or require acquisition of sufficient independently reviewed sites before the principal risk-control study? The former cannot produce a formal safety certificate or restore untouched-test status.

Recommended direction remains the benchmark-compatible empirical mechanism study first, with all previously inspected recordings labeled development material. This recommendation is **not approval**. The easy-degradation ceiling stays at the user's 2%; other statistical tolerances must not be inferred from that number.

New-source permission, annotation review and untouched-test eligibility are separate decisions. Approval of a task cannot clear the DUT duplicate-annotation quarantine or make its two sites into many independent scenes. CREATE access is optional for a small local comparison, not a prerequisite for answering the scientific choices.

## Resume Without Losing Work

After the decisions are supplied:

1. Write a new versioned protocol containing the approved choices and current source/code bindings. Preserve this draft and all historical exposure evidence.
2. Run the contract preflight, then a local cost-estimation pilot using the existing arm64 backend, explicit threads and zero DataLoader workers. Do not retrain models simply to repeat unchanged runtime checks.
3. Fit the prespecified causal/public forecasters and out-of-fold cost heads on assigned training data; compare deferral, ridge and neural heads on identical examples and candidate forecasts.
4. Run independent, scene-uniform and joint intervention controls, including the registered matched-count diagnostic. Select on development only, retain negative results and report the actual independent support.
5. Use approved independent calibration/confirmation data for any formal claim. If none are available, keep that claim unestablished rather than reusing development results as confirmation.

All commands, interfaces and known limits are in the [local/CREATE runbook](local_create_runbook_zh.md). Further architecture modules or repetitive reports will not remove the present decision blocker. No real model has been newly trained, deployed or claimed superior during this continuation. Stage5C and SMC remain disabled. The goal is blocked pending the missing scientific choices, not completed.
