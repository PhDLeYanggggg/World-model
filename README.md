# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling.

The question behind the project is simple:

> If I can see a scene, the agents in it, their recent motion, and their local interactions, can I predict what happens next more reliably than strong causal motion baselines?

I started this repo to answer that question carefully, not just to collect a nice-looking demo. The work here includes the models that improved results, the ones that failed, the leakage checks, the safety rules, and the notes that keep me honest about what the evidence does and does not prove.

## Current Evidence Status

I am running a [matched visual start-information comparison](outputs/publication_readiness_2026_09/source_visual_start_decision.md)
to test whether past pixels add value beyond geometry and image-coverage masks.
The fixed matrix has 30 neural fits and 60,000 updates, with source-only models
shared across the two held-fit sites. Thirty-two focused tests pass, including
exact checkpoint continuation and matched sampling between the image arms.
The input audit finds supported, temporally varying past crops, but does not
establish that body-state cues are visible at the model's 32 x 32 resolution.
Training is still in progress; no forecasting or deployment gain is claimed.
[Reproduction and evidence boundaries](outputs/publication_readiness_2026_09/source_visual_start_v1/reproducibility.md).

I have completed the [source-supported start-information study](outputs/publication_readiness_2026_09/source_start_probe_v1/conclusions.md):
45 classifier fits, including 15 small neural models and 15,000 updates. SDD adds
22,374 supervised stationary-history windows, but none of the fixed models shows
positive probability transfer in both ETH and Hotel.

The important finding is a misleading partial improvement. SDD-trained models
look better on Hotel against the opposite-scene prior, yet all are worse than
simply predicting the SDD start rate. Their overall predicted rate is closer to
Hotel's; their varying per-agent probabilities add error on average. On ETH,
source-only and mixed training both worsen the matched main-only models. This is
not evidence for safer switching or better trajectories, so I am not deploying it.

All 45 prediction replays are exact, completed resume preserves 166 artifacts,
and 17 focused tests pass. The main 8-to-12 forecasting task and sealed roles
remain unchanged. The next question is whether actual past visual state cues
add information beyond this prevalence shift, not whether another threshold
can rescue the same exposed examples.

I have completed the [conditioning comparison](outputs/publication_readiness_2026_09/unit_frame_training_v1/report.md):
27 new neural fits, three seeds, three exposed physical-site folds and 162,000
updates, with nine verified previous controls. The main task and evaluation
metric are unchanged: eight observed annotation steps, twelve predicted steps,
and equal-site past-normalized ADE.

| Geometry model | Gain over constant velocity | Safe positive fits |
| --- | ---: | ---: |
| Previous source-trained control | -0.805% | 0/9 |
| Reconstructed unit-frame inputs | -0.687% | 0/9 |
| Unit-frame inputs and radius-scaled output | -2.489% | 0/9 |
| Same output with internal-coordinate loss | -185.777% | 0/9 |

The [gradient audit](outputs/publication_readiness_2026_09/normalization_response_v2/report.md)
identified weak responses to stationary-to-moving examples, and I repaired a
unit-dependent threshold in the input features. But balancing gradients did
not produce better forecasts. The internal loss permits large errors after
mapping predictions back to the evaluation coordinates. Inputs alone show a
small Hotel improvement, but still harm easy cases and do not generalize across
the three sites. I am not deploying any of these models.

All 27 prediction replays are exact; completed resume preserves 82 artifacts
without training updates. Twenty-four focused tests pass. The intervals and
oracle analyses are exploratory, not independent confirmation. The next step
is to separate useful causal cues from decoder-induced harm, not to expand this
failed loss into a larger model or select thresholds on exposed evaluation data.

I have completed the [source-mechanism controls](outputs/publication_readiness_2026_09/sdd_auxiliary_mechanism_v1/report.md):
54 additional neural fits and 270,000 updates, alongside 54 verified previous
fits. The new controls separate useful SDD supervision from shorter main-task
training. All new checkpoints replay exactly; completed resume preserves 166
artifacts and adds no updates. The full run took about 2.33 hours locally.

The result narrows the interpretation of the earlier SDD improvement. Shorter
main-task training already reduces error, and shuffled SDD labels also help.
Correct SDD labels do not show a stable extra benefit over shuffled labels:

| Input | Correct vs shuffled source gain | Descriptive site interval |
| --- | ---: | --- |
| Geometry | +0.118% | [-0.126%, +1.897%] |
| Coverage masks | -0.015% | [-0.154%, +0.155%] |
| Past RGB | +0.058% | [-0.170%, +0.272%] |

These are comparisons between neural controls, not gains over constant velocity.
Every schedule/input average still loses to CV. Three new individual fits have
tiny positive gains, but none preserves easy cases; there is no new deployable
model. The intervals describe only three already-exposed physical sites.
A [source audit](outputs/publication_readiness_2026_09/sdd_auxiliary_mechanism_v1/source_support.md)
also identified the stationary normalization mismatch tested above. That repair
did not suffice for useful forecasting. I keep the primary task and
sealed evaluation roles unchanged, and do not treat these results as independent
confirmation or a successful world-model contribution.

I have completed the [SDD auxiliary-training comparison](outputs/publication_readiness_2026_09/sdd_auxiliary_v1/conclusions.md):
54 fresh neural fits, three seeds, three physical-site folds and 324,000 optimizer
updates. The original 40 SDD training videos supply 229,333 eligible windows;
the ETH/UCY task, all 11,966 fit windows and sealed evaluation roles stay unchanged.
Training took about three hours on the local arm64 CPU. All 54 checkpoints replay
exactly, and a completed-run resume adds no updates.

SDD pretraining reduces the damage from the neural predictors, but does not make
them better than constant velocity (CV). Positive numbers below mean lower
past-normalized ADE, with equal physical-scene weighting.

| Input | No-SDD gain vs CV | SDD-pretrained gain vs CV | SDD gain vs matched neural control |
| --- | ---: | ---: | ---: |
| Geometry | -1.350% | -0.805% | +0.537% |
| Geometry + image coverage masks | -1.372% | -0.816% | +0.549% |
| Geometry + past RGB | -1.983% | -1.273% | +0.696% |

None of the 54 held-scene fits beats CV or meets the 2% easy-degradation limit.
RGB is also worse than its same-source mask control overall. These are negative
forecasting results, not evidence of a deployable world model. The
[failure analysis](outputs/publication_readiness_2026_09/sdd_auxiliary_v1/failure_analysis.md)
retains absolute easy-case harm, event slices and the limits of the source comparison.
The [design](outputs/publication_readiness_2026_09/sdd_auxiliary_v1_decision.md) and
[input evidence](outputs/publication_readiness_2026_09/sdd_auxiliary_v1/input_evidence.md)
separate training results from data-pipeline checks.

The [working manuscript](outputs/publication_readiness_2026_09/paper_working_draft.md)
now separates what the models learned from what the data pipeline can support.
The one-direction start signal is not a trajectory gain, and the earlier SDD
image/geometry bridge is distinct from the new auxiliary-training comparison. I keep the negative
comparisons visible; independent confirmation and a positive method contribution
are still missing.

I am testing why better-fitting neural forecasts still lose to a strong motion
baseline across scenes. The current task observes eight annotation steps and
predicts twelve. I keep the primary metric, fit cohort and sealed evaluation
roles fixed while changing one factor at a time.

| Question | Evidence | Outcome |
| --- | --- | --- |
| Is the source benefit specifically learned motion transfer? | [54 new mechanism controls, 54 verified previous fits](outputs/publication_readiness_2026_09/sdd_auxiliary_mechanism_v1/report.md) | Shorter main exposure and shuffled source labels explain part of the apparent benefit; correct pairing has no stable extra gain, and no model is safe. |
| Does additional SDD supervision improve transfer? | [54 matched fresh fits](outputs/publication_readiness_2026_09/sdd_auxiliary_v1/conclusions.md) | It improves on the neural controls, but all models remain worse than CV and fail easy preservation. |
| Does balancing tracks and state changes help? | [54 new fits, 18 replayed controls](outputs/publication_readiness_2026_09/track_event_sampling/conclusions.md) | No safe forecasting gain; repeating rare starts does not add independent information. |
| Could better routing rescue the existing predictions? | [Frozen-candidate ceiling](outputs/publication_readiness_2026_09/candidate_headroom/conclusions.md) | Perfect future-informed selection gains 1.63%, or 1.73% with whole-path scaling. These are oracle diagnostics, not model results. |
| Are corrections difficult to fit because of their range? | [54 new fits, 18 controls](outputs/publication_readiness_2026_09/residual_range/conclusions.md) | Transformed targets improve training fit but worsen held-scene accuracy and easy-case harm. |
| Does a past-only coordinate frame improve transfer? | [36 new fits, 18 controls](outputs/publication_readiness_2026_09/past_frame/conclusions.md) | Static-input rotation consistency improves, but primary gains remain -0.86% and -0.89% versus CV; easy preservation fails. |
| Do past motion features predict a recorded start at all? | [48 fixed classifier fits](outputs/publication_readiness_2026_09/motion_start_information/conclusions.md) | Tree models show a Hotel-to-ETH signal (AUROC 0.81-0.82), but reverse transfer stays near chance. This is not a trajectory gain. |

These studies reuse some controls and the same exposed fit scenes; they are not
independent confirmations. No new model is promoted. The next useful evidence
needs transferable cues for starting, stopping and turning, rather than another
threshold sweep over the same weak predictions. The detailed reports retain
negative seeds, absolute harm, runtime and reproducibility checks.

The latest probe suggests that observed motion is not wholly uninformative, but
the signal is one-directional and supported by few agents. I am not using its
best score to justify another deployment or a world-model success claim.

The SDD [past-image and trajectory interface is now joined](outputs/publication_readiness_2026_09/sdd_multimodal_bridge/conclusions.md)
across the original 40 training recordings, including their later frames. All
5,074 fixed query histories pass identity and past-frame checks; sampled image
replays and restart checks agree exactly. This fixes a real limitation of the
old first-64-frame cache. It is not a training result. The input audit also shows
that 38.7% of sampled annotation boxes have a projected short side below eight
pixels, so available visual detail remains a limitation to test, not assume away.

I have now built a [past-only SDD data bridge](outputs/publication_readiness_2026_09/sdd_step_bridge/conclusions.md)
for the eight-observed/twelve-predicted model interface. It covers the original
40 training videos, keeps future labels separate, and preserves inputs even when
future labels are incomplete. At a diagnostic stride of 12 raw frames it indexes
229,333 overlapping windows; this is not an independent sample count or a model
score. Full-index verification and resume checks pass. The separately approved
train-40, stride-12 auxiliary experiment above now uses a complete image cache;
the earlier bridge itself remains diagnostic. The main ETH/UCY evaluation is unchanged.

I have now [counted the state-change support in all local SDD annotations](outputs/publication_readiness_2026_09/sdd_state_support/conclusions.md).
Large window counts conceal a much smaller set of relevant trajectories: at a
diagnostic stride of 12 raw frames, nearly 250,000 complete pedestrian windows
contain only 78 recording-local tracks matching the fixed static-to-movement
proxy. Turns and stops have broader support. This points toward controlled
event/track sampling, not simply multiplying overlapping windows. The census
itself trained nothing. The subsequent train-only auxiliary trial above tests
one fixed sampling interval; no forecasting gain is claimed from the census.

I have connected the repaired SDD video paths to a
[past-image reader with explicit missing support](outputs/publication_readiness_2026_09/sdd_past_images/conclusions.md).
It keeps partial observations and short histories, distinguishes image extent
from suspected black padding, and preserves occlusion flags. The fixed-prefix
check covers all 60 recordings; 2,074 sampled crops replay exactly. These are
input checks, not better forecasting scores. The padding mask is inferred, and
the later SDD training admission is explicit and separately registered.

I found and repaired two problems in the local SDD video input path before
expanding training: compressed video pixels do not share the annotation image
size, and ten Nexus clips are paired with the wrong annotation names. The
[full source audit](outputs/publication_readiness_2026_09/sdd_media_alignment/conclusions.md)
decodes all 522,497 frames and preserves an explicit, hashed correspondence map.
This fixes diagnostic image access, not forecasting performance. The reader above
now handles partial support. The auxiliary protocol has since been approved and
run; actor visibility and full semantic alignment are not certified by pixel replay.
The existing ETH/UCY results and sealed evaluation roles are unchanged.

I have completed the [native-resolution motion study](outputs/publication_readiness_2026_09/spatial_motion/conclusions.md):
36 matched neural fits test whether original image detail and localized motion
grids improve forecasting. None beats the training-selected constant-velocity
baseline; native detail is slightly worse overall than the lowpass grid. All
checkpoints replay exactly, and easy-case errors remain unacceptable. I do not
promote these models or count clearer inputs as better predictions.

The most useful next step is broader independent evidence of starting, stopping
and turning, not another threshold sweep on the same few people. I have also
[inventoried 60 local SDD videos](outputs/publication_readiness_2026_09/sdd_media_inventory/report.md).
The follow-up above goes beyond readable headers and exposes real correspondence
defects. The separately registered SDD trial above is now complete. It neither
certifies full semantic alignment nor changes the primary metric.

I have completed the [past image-motion comparison](outputs/publication_readiness_2026_09/observed_motion_v2/conclusions.md):
54 real neural fits test quality controls, motion magnitude and motion direction
on all 11,966 fit windows. Correcting moving crops and adding direction reduces
some damage from an ADE-trained network, but none of the models beats CV or
preserves easy cases. Every checkpoint replays exactly; no model is promoted.

The main remaining issue is transferable state-change information. Past image
motion is not the same as knowing whether or where a stationary person will
start moving, and the independent training support is still small. I am keeping
the eight-observed/twelve-predicted task and sealed evaluation roles unchanged,
and retaining negative evidence instead of selecting a favorable scene.

I have completed the [training-objective comparison](outputs/publication_readiness_2026_09/objective_alignment/conclusions.md):
45 matched neural fits separate scene sampling, mean-ADE training and a
baseline-relative harm penalty. Training performance improves substantially,
but every held-scene model remains worse than CV. The loss-only repair fails.
On Hotel, false movement after stationary histories dominates the severe damage;
the native-coordinate diagnostics are negative too. All 45 checkpoints replay
exactly, and no model is promoted.

This narrows my next step to better observed state/direction and independent
start/stop support, rather than more loss weights or fallback thresholds. The
approved eight-to-twelve task, primary metric and sealed evaluation roles stay
unchanged. The [report](outputs/publication_readiness_2026_09/objective_alignment/conclusions.md)
keeps the easy failures, three-seed results and limits of the three-site evidence.

I have completed a [broader matched visual forecasting study](outputs/publication_readiness_2026_09/offline_visual_forecast/conclusions.md)
on all 11,966 approved fit windows: 36 neural fits, three seeds, three physical
scenes and 72,000 optimizer updates. Geometry, coverage masks, current RGB and
past RGB all remain below the training-selected CV baseline, by 0.58%, 0.57%,
0.66% and 0.74% respectively. Training loss improves, but cross-scene prediction
does not. All checkpoints replay exactly; no new model is deployed.

I use standard offline annotated histories with interpolation provenance
disclosed, not a strict real-time sensor-causality claim. Zara recordings stay
in one scene group, missing images do not remove rows, and the approved
eight-to-twelve task and past-normalized ADE are unchanged. Separate development,
calibration and confirmation roles remain unused in this study.

The new diagnosis narrows the next step: 31 stationary-history source IDs still
account for 89.37% of the equal-scene baseline error. Even a perfect retrospective
CV/neural chooser captures only 0.19--0.31% with these fixed predictions. I need
better supported motion predictions before more fallback tuning, not a larger
version of the same weak candidate. This is fit-only negative evidence, not proof
that vision cannot help. [Results and limitations](outputs/publication_readiness_2026_09/offline_visual_forecast/conclusions.md).

I have now [built masked past-image inputs](outputs/publication_readiness_2026_09/zara_masked_images/conclusions.md)
after repairing the Zara video-coordinate mapping. The reader retains12,098
eight-step histories, including1,935 with partial edge crops. It keeps the real
visible pixels and an explicit coverage mask instead of throwing away the whole
image. An independent rebuild reproduces every cache array exactly. This is
an input repair, not a new forecasting result.

The same audit exposed an important limit: most stored Zara past trajectories
were interpolated using later annotation controls. An offline trajectory
benchmark and strictly real-time sensor observations are therefore different
claims. I have not changed the approved eight-to-twelve protocol, discarded
those rows, or called the source mapping physical calibration. That observation
definition is now explicitly offline annotated forecasting for the new comparison.

I have followed the appearance failure with a
[fixed-model support control](outputs/publication_readiness_2026_09/appearance_support_control/conclusions.md)
and [six matched neural refits](outputs/publication_readiness_2026_09/appearance_no_camera/conclusions.md).
Clipping camera/context features reduces some Hotel damage, but no treatment
beats CV. Removing the learned camera inputs and retraining has the same tradeoff:
Hotel improves relative to the damaged model, while ETH gets worse. Exact
checkpoint replay confirms the results. Simply fixing a feature range is not
enough to recover useful cross-scene motion.

The strict support rule rejects every cross-scene query and returns CV. I do not
count that as learning or generalization. My next priority is broader verified
past-context support within the approved fit recordings, rather than more
threshold or camera-normalization sweeps on these same few agents.

I have completed the [past-appearance prediction experiment](outputs/publication_readiness_2026_09/past_appearance_probe/conclusions.md):
18 real neural fits compare geometry, one current image and eight observed images
across three seeds and two held fit scenes. None improves on CV after the fixed
confidence gate. The networks fit one scene but transfer poorly, and image inputs
can make them more confident without making their trajectories better. All final
checkpoints replay exactly; this is a negative predictive result, not a runtime
failure or a new deployment.

The small stationary subset contains only five ETH and 26 Hotel agents. On the
ETH-to-Hotel direction, 78.52% of held rows exceed the fixed +/-10 standard-score
range under ETH-only normalization. I am separating this camera/context
shift from actual visual information before scaling up. A start probability is
not enough to justify replacing a strong motion baseline.

The preceding [moving-image correspondence checks](outputs/publication_readiness_2026_09/past_motion_comparison/conclusions.md)
support the native-index input path, but do not certify pose, physical timing or
scale. All 32 incomplete image-window rows remain visible in the new experiment;
none is silently dropped to improve the score.

I have completed a [controlled point-forecast comparison](outputs/publication_readiness_2026_09/conditional_ade_probe_refined/conclusions.md):
keep the same trees and features, but replace their average trajectory with a
conditional geometric median matched to the distance error. This removes false
movement on ETH and reduces Hotel damage substantially, yet none of the18 fixed
settings beats CV. The ETH result is an exact fallback, not a learned gain.
One tiny numerical convergence gap is retained explicitly. Better point decisions
help, but the current context still does not support useful cross-scene motion.

I have now tested whether the stationary-start failure is mainly a coordinate
precision issue. The [source and quantization checks](outputs/publication_readiness_2026_09/stationary_label_resolution/conclusions.md)
rule out printed rounding, and a hidden constant-velocity path inside half-pixel
rounding cells cannot explain 98.30% of ETH or 97.21% of Hotel error on this subset.
These are fit-only diagnostics, not new training or proof of physical movement.
The forecasts also remain weak on larger changes. My next modeling step needs
useful directional information, not just another threshold or precision fix.

I have followed the stationary-start study with a
[static-scene and direction experiment](outputs/publication_readiness_2026_09/stationary_scene_probe_v2/conclusions.md).
Supplied obstacle context helps some start-probability scores, but it does not
recover useful trajectories: all 36 corrected trajectory regressors remain worse
than CV, and a fixed confidence gate gives zero or negative gain. I also found
and repaired a shared-corner geometry bug; the two small positive results from
the original version disappear after that repair. Both versions are retained.
These are small fit-only classifier/regressor experiments, not a new neural
world model. The missing evidence is useful motion prediction, not just start AUC.

I have now traced the stationary-start failure back to the original annotations
and completed a [fit-only context study](outputs/publication_readiness_2026_09/stationary_start_probe/conclusions.md).
The 365 stationary windows are just 31 agents and 45 runs at two sites, not
365 independent starts. All 24 initial neighbor-context classifiers were worse
than a training-only prior. Pooling the context into fewer features recovered
a signal from Hotel to ETH, but not in the reverse direction. I keep that local
signal and the failures visible; neither establishes better trajectories or a
new deployment. The primary task remains eight observed to twelve predicted
native steps, with raw-frame t+50 as a supplement.

I have completed the [cost-sensitive deferral control](outputs/publication_readiness_2026_09/8to12_deferral_v7/conclusions.md)
on the frozen v7 predictors: 24 new routing heads, three seeds, identical
out-of-fold feature rows and no retrospective threshold selection. The bounded
predictor has small positive average gains, but every deferrer fails easy-case
preservation. This closes a missing method comparison without establishing a
new deployable model. It also strengthens the case for improving the candidate
forecasts before spending more effort on routing thresholds.

My latest [paired study](outputs/publication_readiness_2026_09/8to12_residual_pair_v7/conclusions.md)
is complete: learn corrections from an exact CV initialization, with and without
a past-motion amplitude bound. Both versions retain the v6 data, primary metric,
loss, budgets and policy rules. All three seeds, 24 forecasting fits, six neural
cost heads and six ridge controls finished; the training was not shortened.

The repair changes mean uncontrolled normalized-ADE gain from -0.599% for the
CV-skip to +0.062% for the bounded version. That small positive average comes
with unacceptable easy-case damage without selection. The guarded development
gains are only +0.002495%, +0.007710% and +0.000519%. I treat this as evidence
of reduced drift, not a new deployable model or a submission-quality advantage.
The [completed raw50 and same-count controls](outputs/publication_readiness_2026_09/8to12_residual_pair_v7/supplement_conclusions.md)
show no extra benefit from joint routing: all CV-skip differences are zero,
and the bounded arm's three nonzero ADE differences favor independent routing.

A [fit-only geometric check](outputs/publication_readiness_2026_09/8to12_residual_pair_v7/fit_bound_headroom.md)
quantifies this constraint before interpreting its results. Exactly stationary
pasts account for 73.3% of fit CV error under the unchanged normalization. The
bound cannot correct those starts; even a label-aware per-step oracle has only
4.09% average fit-window headroom. That is an optimistic capacity diagnostic,
not a learned result, a development estimate, or a reason to drop difficult rows.

The preceding complete three-seed comparison covers both a local Transformer and the
EqMotion author core, adapted to one fixed prediction head. It includes 24
10,000-update forecasting fits, six 1,000-update neural cost heads and six ridge
controls. Both models use the same observed support, loss and sample/update
budget, but not the same parameter count or compute cost.

| Model | Seed 17: primary ADE gain vs CV | Seed 29 | Seed 43 | Development selection |
| --- | ---: | ---: | ---: | --- |
| Transformer | -5.74% | -7.69% | -6.70% | CV in all seeds |
| EqMotion, fixed K=1 | -14.12% | -8.47% | -14.08% | CV in all seeds |

These are negative development results, not a new deployment. Past-only input
conditioning allows the full EqMotion budget to finish without the earlier
overflows, but numerical stability is not forecasting success. The
[paired results](outputs/publication_readiness_2026_09/8to12_public_predictors_v6/paired_predictor_comparison.md)
keep all seeds, easy-case damage and stronger causal alternatives visible.
The diagnostic candidate/CV oracle has less than 0.61% headroom on the fixed
primary metric, so additional threshold search cannot yield a 5% improvement
from these unchanged forecasts.

An [error decomposition](outputs/publication_readiness_2026_09/8to12_eqmotion_v6/error_scale.md)
locates most damage in the 10.9% of complete queries whose past-motion scale is
at the numerical floor. I retain those cases and the original metric. EqMotion
does beat the stronger causal alternative on Students03 in native coordinates
by 0.94--2.32%, but not on Students01. This recording-specific signal does not
override the failed primary result or establish cross-scene generalization.
The [frozen decision](outputs/publication_readiness_2026_09/conditioned_context_v6_decision.md)
specifies what changed and what stayed fixed.

The [completed raw50 and same-count supplements](outputs/publication_readiness_2026_09/8to12_public_predictors_v6/supplement_conclusions.md)
do not reverse that conclusion. Transformer's joint and independent policies
choose the same agents; EqMotion changes a small number of decisions without a
stable gain. All joint policies have negative raw50 ADE improvement. I report
these controls to test the proposed mechanism, not to replace a failed primary
result with a favorable secondary number.

During the preceding run I also traced an upstream
Students01 packaging issue: 20-point fragments preserve timestamps and positions
but change identities and remove short/tail tracks using later availability.
I preserved the completed 10,000-update fit, stopped before scoring that context,
and rebuilt the development source with continuous identities. The
[new protocol](outputs/publication_readiness_2026_09/continuous_context_v5_decision.md)
preserves that repair; it does not rename historical data as independent confirmation.
An [execution-only optimization](outputs/publication_readiness_2026_09/public_predictor_v4_execution_decision.md)
skips unused output heads while preserving selected outputs and gradients in
CPU/MPS tests. This remains a K=1 adaptation, not published best-of-20 EqMotion.
The replacement full fit matches the preserved model's parameters and complete
loss sequence exactly. Three v5 EqMotion fits finished, but the next fold
encountered float32 overflow. I reproduced it separately and completed that fold
on CPU at the unchanged 10,000-update budget. A later seed failed on both devices,
so v5 remains an incomplete comparison rather than a selectively reported success. The
[failure record](outputs/publication_readiness_2026_09/8to12_public_predictors_v5/nonfinite_fit_diagnosis.md)
keeps the evidence visible. That v5 failure is separate from the completed v6
paired experiment above, not silently overwritten by the numerical repair.

My primary task is now **eight observed steps to twelve predicted steps**, with
raw-frame `t+50` retained as a separate supplement. I am prioritizing a focused
paper on baseline-relative joint intervention: when a neural forecast is worth
using, and whether interacting agents should switch together. The
[research route](outputs/publication_readiness_2026_09/research_route_decision.md)
sets out the contribution, falsifiable controls and publication boundaries.

My first **development-only** ETH/UCY experiment is complete: three seeds,
12 forecasting fits and three neural cost-head fits, each with 1,000 updates.
It uses 11,966 fit windows, three physical-scene crossfit folds and UCY University
for development. All three seeds selected the constant-velocity floor. Without
fallback, the neural model worsened the primary normalized ADE by 7.09%, 7.90%
and 7.25%. The [complete results](outputs/publication_readiness_2026_09/8to12_development_v1/results.md)
retain native-coordinate errors, easy-case damage and the negative controls.
These are real training results on historically exposed development data, not
independent confirmation or a calibrated safety guarantee.

A fit-only audit found highly concentrated squared target energy after
past-based normalization. I completed a second three-seed experiment changing
only the forecasting loss to Smooth-L1. Average degradation versus CV fell from
7.41% to 0.25%, but all seeds still selected the floor. Joint selection added no
measured benefit. The [paired loss ablation](outputs/publication_readiness_2026_09/8to12_robust_v2/robust_loss_comparison.md)
also shows why gains over CV alone are insufficient: the better damped-velocity
baseline explains most of the favorable native-coordinate result. This is a
useful training repair, not a successful intervention method or deployment upgrade.
I have not changed the primary metric after observing which one looks better.
The [Chinese research roadmap](outputs/publication_readiness_2026_09/publication_route_zh.md)
explains the experiment order and what would justify a later journal extension.

The protected selector remains the historical reference implementation. It starts from causal motion baselines and switches only when the expected gain is large enough and the estimated easy-case risk is low enough. I am not currently treating its external results as independently validated deployment evidence.

During the September 2026 publication audit, I found byte-identical recordings under different dataset paths. The Stage35/37 validation and test sets share 47,223 cached windows, and the later Stage43/44 split has train/test duplication as well as inherited teacher-training exposure. Stage37 and the original Stage44 code also used test metrics to rank model variants. These issues mean the external claims need a clean rerun. The [recording audit](outputs/publication_readiness_2026_09/recording_lineage_audit.md) and [original split audit](outputs/publication_readiness_2026_09/stage35_recording_lineage_audit.md) document the evidence.

For traceability, these are the **historically reported Stage37 numbers, not corrected confirmatory results**:

| Slice | Result |
| --- | ---: |
| Overall improvement | +13.48% |
| Raw-frame `t+50` improvement | +8.46% |
| Hard/failure improvement | +15.54% |
| Easy-case degradation | 0.041% |
| `t+50` bootstrap CI | [+7.69%, +9.15%] |

The table is retained so that older reports remain interpretable. Its bootstrap interval does not account for duplicate recordings or test-based model selection. I have repaired the WorldCore selection path to use validation only and added a training preflight that rejects the unchanged legacy caches. This is a protocol repair, not a new model improvement; the historical weights have not been replaced.

I have also rebuilt a recording-centric reader directly from the original positions, without inherited teacher outputs. It keeps past inputs separate from future labels, groups duplicate dataset packaging, and requires exact target timestamps. The [rebuild and causal-access checks](outputs/publication_readiness_2026_09/causal_recording_checks.md) are engineering evidence, not a new benchmark result. The new development protocol is frozen; an independent confirmation protocol remains pending.

The next method asks a more specific question: when several agents may switch away from a baseline, should those decisions be made together? I have implemented a [joint intervention prototype](outputs/publication_readiness_2026_09/joint_intervention/method_and_checks.md) with explicit predicted-risk and intervention budgets. Its optimizer and coordinate handling pass targeted checks, but it has not yet demonstrated better prediction or calibrated real-world risk. Joint trajectory modeling already has substantial prior work; the contribution will need to come from the matched experiments, not the presence of a joint optimizer.

I have added an [exact-coverage control](outputs/publication_readiness_2026_09/matched_coverage/method_and_limits.md) to test that distinction: independent and joint policies must switch the same number of agents in each scene query. It prevents a policy that simply switches less from being credited with better coordination. The completed v6 diagnostic finds no stable joint gain; this is not a deployable policy or a risk guarantee. [Regression-deferral prior work](outputs/publication_readiness_2026_09/joint_intervention/deferral_and_coverage_prior_work.md) also makes clear that learning when to use another predictor is not a novelty claim by itself.

I also have a [cost-sensitive deferral control](outputs/publication_readiness_2026_09/deferral_control/method_and_limits.md) derived from that prior work. It learns from the same out-of-fold predictions as my gain/harm head and retains the size of each forecasting error instead of learning only which predictor wins. Its fitting, recovery and leakage checks work on synthetic examples. The real v7 comparison is now complete and reported above; no tested setting combines positive gain with the required easy-case preservation.

The control is connected to the [development comparison](outputs/publication_readiness_2026_09/deferral_development/implementation_and_limits.md). Both heads use identical training queries and causal features, and every arm sees the same candidate forecasts before labels are opened. I report actual intervention rates separately: an unconstrained deferral gate does not share M3W's risk budget just because it shares its predictions. The completed real comparison remains development evidence, not independent confirmation.

For the clean rerun, I keep data roles and learned-artifact provenance in a [hash-bound experiment contract](outputs/publication_readiness_2026_09/experiment_contract/implementation_and_limits.md). It checks upstream teacher exposure and freezes the evaluated model family before calibration or confirmation. The original independent-study draft remains unapproved. A separate versioned development protocol now permits fitting without inventing calibration or confirmation roles for previously explored data.

I am also checking what the other local datasets can genuinely add. The [external source review](outputs/publication_readiness_2026_09/external_source_audit/source_review.md) distinguishes synchronized scenes from isolated tracks, repeated controlled trials and duplicate representations. More trajectory points do not automatically mean more independent evidence; none of these sources has been promoted to a new confirmation set or used to claim a fresh gain.

The local [CITR diagnostic reader](outputs/publication_readiness_2026_09/citr_causal_intake/implementation_and_limits.md) now preserves 95,648 original positions across 38 controlled clips, with separate pedestrian and vehicle identities and an exact map back to each CSV row. This gives me synchronized mixed-agent examples for future mechanism checks. All clips still belong to one physical site; they are not 38 new independent test scenes, and I have not used this conversion to claim a forecasting gain.

I have also brought in the author's unfiltered [DUT campus annotations](outputs/publication_readiness_2026_09/dut_causal_intake/implementation_and_limits.md): 457,686 positions from 28 clips at two locations, with exact source-row checks. The audit caught two pedestrian IDs with identical 145-frame trajectories in one clip, which is now flagged for review before any formal use. It also exposed a raw-coordinate ambiguity in the source documentation. These data extend the diagnostic material available to the project, but they are not yet an approved independent benchmark, and no forecasting gain is claimed from their acquisition.

I now enforce that distinction in the [data-admission path](outputs/publication_readiness_2026_09/intake_admission/implementation_and_limits.md): approving an experiment cannot silently clear an annotation quarantine or pending source-use review. The checks follow the underlying audit and keep unresolved data out of fitting and evaluation. They preserve the evidence needed to investigate a defect; they do not substitute for permission, independent data or a scientific result.

The rebuilt reader connects to a [resumable forecasting and cost-learning backend](outputs/publication_readiness_2026_09/supervised_backend/implementation_and_limits.md). It learns benefit and harm from predictions made outside each producer's training fold. The [development evaluator](outputs/publication_readiness_2026_09/development_evaluation/implementation_and_limits.md) compares five controls on the same predictions and keeps agents with missing future labels in the decisions. The first real three-seed comparison is now complete and negative. Independent confirmation and a useful scene-level risk guarantee remain open.

To avoid comparing only against my own networks, I have connected a [version-pinned EqMotion core](outputs/publication_readiness_2026_09/public_baselines/compatibility_and_limits.md) to the causal reader and training path. The current adapter emits one fixed head and is explicitly a K=1 adaptation, not a reproduction of the paper's best-of-20 result. Its complete real comparison is reported above. The published model's preprocessing, sampling budget and checkpoint selection need the same scrutiny as my own code.

I now have a [frozen-policy risk-screening path](outputs/publication_readiness_2026_09/risk_calibration/implementation_and_limits.md), but implementation is not a safety guarantee. The current rebuilt collection contains only six physical-scene groups, all previously used during development. Even an optimistic calculation shows that the present conservative bound would be too wide for a useful small-risk claim. My [statistical assumption review](outputs/publication_readiness_2026_09/joint_intervention/statistical_assumptions_and_claims.md) checks why existing uncertainty methods do not automatically remove that limitation. I keep prediction-set coverage, improvement over a baseline and physical safety separate; duplicating windows cannot close the evidence gap.

The [final comparison path](outputs/publication_readiness_2026_09/confirmation_evaluation/implementation_and_limits.md) keeps every training seed visible and freezes the compared policies before final labels are read. It reports scene-level uncertainty, per-seed easy-case damage and matched joint-control comparisons without choosing a winner on the final set. The complete connection has been exercised on synthetic data; it does not replace the still-pending real independent experiment.

The [neural benefit/harm head](outputs/publication_readiness_2026_09/neural_cost_head/implementation_and_limits.md) trains on the same held-fold inputs as the ridge control, so neural capacity is not confounded with different examples or candidate forecasts. CPU/MPS recovery is checked. Both the synthetic example and the completed real MSE experiment select the baseline; I have not established a real-data advantage for this head.

## What The System Looks At

The current M3W pipeline works with dataset-local top-down trajectories. It uses information that would be available at inference time:

- recent agent history;
- speed, acceleration, heading, curvature, and stop/go behavior;
- neighbor density and interaction signals;
- train-only scene or goal context when that context is legally available;
- causal baseline rollouts;
- dataset, scene, horizon, and domain metadata;
- risk heads for failure, gain, harm, and fallback decisions.

I also maintain a neural track with Transformer dynamics, JEPA-style representation learning, hybrid heads, waypoint prediction, and protected residual policies. Guarded selection, causal history windows, full-waypoint structure, domain-aware routing, and safety floors are the most promising routes in the historical experiments. Their external gains remain exploratory until the clean evaluation is complete; neither the selector nor the neural branch has earned a new deployment claim from this audit.

## What This Repo Is For

This repository is a research record. The most important rule in the project is that a result has to survive the boring checks: no future leakage, no test endpoint goals, no central-velocity shortcuts, no easy-case damage hidden inside aggregate gains, and no metric claims without calibration.

For a quick orientation:

| File or directory | What to read it for |
| --- | --- |
| [`README_RESULTS.md`](README_RESULTS.md) | Detailed results ledger and current evidence boundaries. |
| [`README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md`](README_M3W_WORK_ATTEMPTS_FAILURES_SUCCESSES_ZH.md) | Chinese long-form summary of routes tried, failures, causes, and successes. |
| [`research_state.json`](research_state.json) | Machine-readable snapshot of the current project state. |
| `outputs/m3w_neural_v1/` | Neural world-model reports and model-card style summaries. |
| `outputs/stage42_long_research/` | Cross-domain safety, replay, full-waypoint, and paper-claim evidence. |
| `outputs/stage43_latent_state/` | Latent-state, graph/history/context, and reviewer-style validation reports. |

Large datasets, caches, checkpoints, videos, images, third-party data, and local virtual environments are intentionally kept out of git.

## What I Am Not Claiming

M3W is not a true 3D world model yet. It is not a foundation world model. SDD results are pixel-space unless calibration is verified. External results are dataset-local unless their geometry is verified. `t+50` and `t+100` are raw annotation-frame horizons, not seconds. Self-audited or inferred labels are not human gold labels.

Stage5C latent generative execution has not been enabled. SMC has not been enabled.

The current claim is narrower: this repo contains a protected 2.5D multi-agent world-state research system and an active neural dynamics track. Its historical external evaluation has identified independence failures that I am repairing before making new generalization or deployment claims. This external audit does not establish the status of every SDD experiment.

## Running Locally

On Apple Silicon, training should use the arm64 PyTorch environment:

```bash
.venv-pytorch/bin/python
```

Focused checks for the new data and evaluation path:

```bash
.venv-pytorch/bin/python -m pytest tests/test_m3w_deferral_development.py -q
.venv-pytorch/bin/python -m pytest tests/test_m3w_cost_sensitive_deferral.py tests/test_m3w_matched_coverage.py tests/test_m3w_citr_recordings.py tests/test_m3w_confirmation_evaluation.py tests/test_m3w_risk_calibration.py tests/test_m3w_eqmotion_adapter.py tests/test_m3w_development_evaluation.py tests/test_m3w_supervised_intervention.py tests/test_m3w_external_source_audit.py tests/test_m3w_experiment_contract.py tests/test_m3w_joint_intervention.py tests/test_m3w_causal_recordings.py tests/test_m3w_recording_lineage.py tests/test_stage44_worldcore.py
```

The legacy full suite (`python -m pytest tests`) includes integration training and can rewrite reports in the working directory. It is not yet an isolated, read-only smoke test; preserve existing experiment outputs before running it. The [local/CREATE runbook](outputs/publication_readiness_2026_09/local_create_runbook_zh.md) records the verified environment and recovery checks.

Training scripts are written around checkpointing, heartbeat logs, resume support, CPU/MPS-safe execution, and single-process dataloading.

## Next Step

I am developing this work toward a CVPR 2027 submission on when neural motion predictions can safely improve a strong baseline. The next study focuses on baseline-relative risk and joint intervention across agents. My [research direction and evidence audit](README_M3W_INNOVATION_AND_CVPR2027_ZH.md) explains the proposed contributions, the limitations of the current experiments, and the remaining comparisons. In particular, the latest WorldCore architecture ranking used test metrics, so those results remain exploratory until independently confirmed.

The eight-observation/twelve-prediction development task is frozen and the
matched three-seed v6 training and fixed-forecast supplements are complete,
without changing the selected floor. Independent calibration and
confirmation remain unresolved; the development runs cannot supply those claims. The
[continuation record](outputs/publication_readiness_2026_09/continuation_handoff.md)
and [runbook](outputs/publication_readiness_2026_09/local_create_runbook_zh.md)
record the completed protocol, checkpoint paths and recovery commands. The next
prospective test should address near-stationary motion and scale handling, not
retune these frozen policies until one secondary metric looks favorable.

The next research step is to make the neural branch contribute something the protected policy does not already provide.

That means:

1. rebuild recording-disjoint evaluation and refit the protected selector within each training fold;
2. promote neural dynamics only if they improve the frozen 8-to-12 task without damaging easy cases; keep raw-frame `t+50` as a separate supplement;
3. keep testing whether scene, goal, graph, and latent context add measurable lift;
4. keep raw-frame and dataset-local claims separate from metric or physical-world claims.

When a route fails, I leave it in the record. When a route works, I want the repo to show exactly where it works, why it is allowed, and what it still does not prove.
