# M3W: Real-World Multimodal Agent-Scene World Model

M3W is my research project on top-down multi-agent world modeling. I study
when a neural forecast adds value over a strong motion baseline, and how to
keep the cases that the baseline already handles well from getting worse.

The long-term aim is a useful agent-scene world model. The current work is
narrower: causal trajectory forecasting, learning intervention gain and harm,
and deciding when a neural prediction should replace a physical baseline.
Combining JEPA, Transformer and other modules is not, by itself, a contribution.

I trained a vehicle-specific alternative to the pedestrian weights in
the Songdo bank. It beats CV on the conservative OpenDD forecast score
at all seven held locations, but using it on every vehicle almost
always fails the original risk rule. It is a candidate, not a replacement.
[Vehicle forecast study](outputs/publication_readiness_2026_09/opendd_vehicle_residual_v1/README.md).

I generated unscored Songdo forecasts from past tracks alone. At a
common radius I can compare additive, same-family and joint choices
without counting the same pair twice. The new search agrees with the
old implementation on the largest Songdo query.
[Forecast bank and search check](outputs/publication_readiness_2026_09/songdo_candidate_bank_v1/README.md).

I built Songdo's eight-step inputs separately from its future label
index. The A and B views share an observed footprint, so I keep them
in the same provisional source group even though no sampled duplicate
was found. [Input pack and source grouping](outputs/publication_readiness_2026_09/songdo_prefix_pack_v1/README.md).

I checked Songdo Traffic as a new external source. Almost every eligible
nonoverlapping window has a fully labelled vehicle pair, but only 124 of
1,992 have complete futures for the whole observed roster. I kept those
two kinds of coverage separate and have not scored a forecasting policy.
[Songdo source and clock audit](outputs/publication_readiness_2026_09/songdo_traffic_intake_v1/README.md).

I also required the neighbor to predict the target's motion in two
already observed steps before using it. This left only 99 matched edits
and did not improve the twelve-step pair forecast over own history.
[Lagged neighbor test](outputs/publication_readiness_2026_09/opendd_peer_lag_v1/README.md).

I tested whether a nearby agent's recent velocity adds more than another
forecast based on the target's own history. With the same edit budget,
both added a few useful choices in hindsight. The neighbor version did
not reliably beat its self-history control when used without hindsight.
[Matched neighbor forecast test](outputs/publication_readiness_2026_09/opendd_peer_transient_v1/README.md).

I checked what the joint selector changes when it has exactly the same
forecast choices and edit budget as an additive selector. It often changes
which people get an edit, but did not reliably improve relative motion.
Even choosing among the three frozen outputs with hindsight adds only
0.044 percentage points on the conservative score.
[Matched-budget attribution](outputs/publication_readiness_2026_09/opendd_pair_attribution_v1/README.md).

I used a second detector to test when a P-DESTRE track is reliable.
Agreement did not beat simple confidence at the same track count.
Averaging the two detectors' past positions did reduce relative
trajectory error on the same matched tracks.
[Track agreement and center averaging](outputs/publication_readiness_2026_09/pdestre_track_agreement_v1/README.md).

I ran two frozen detectors on real P-DESTRE video. The first scoring
pass exposed a box-width/height discrepancy. Swapping those fields
restored many reference matches without changing the camera output,
but neither detector recovered a completely scorable online roster.
[Camera and box-format comparison](outputs/publication_readiness_2026_09/pdestre_camera_support_v1/README.md).

I checked P-DESTRE for a new external forecasting study. Its video
streams and eight-to-twelve-step labels are usable, but shared
identities connect all 75 videos. I treat it as one linked source,
not as 75 independent tests.
[Video access and source audit](outputs/publication_readiness_2026_09/pdestre_source_intake_v1/README.md).

I trained two AV2 scores to rank forecasts that pass both accuracy and
risk. The new ranking loss did better than its own classifier when an
easy reference was known, but it did not beat the existing score.
Neither new model qualified for release on the calibration cities.
[Direct qualification and ranking test](outputs/publication_readiness_2026_09/av2_release_rank_v1/README.md).

I checked whether the AV2 release failures were just a threshold
problem. At the same nine releases that passed calibration, only
three of the highest-ranked forecasts passed on the held cities.
All nine slots could have been filled by qualifying forecasts.
Even knowing which views had an easy reference did not fix the ranking.
[Calibration and ranking diagnosis](outputs/publication_readiness_2026_09/av2_release_transport_v1/README.md).

I checked whether requiring an easy reference before ranking would make
the AV2 forecasts safer to release. The main gate did not change which
forecasts were released. A stricter gate found more cases with a usable
reference, but those cases brought back observed harm at the same
intervention count. I did not keep either rule for deployment.
[Matched support gate test](outputs/publication_readiness_2026_09/av2_separate_support_gate_v1/README.md).

I trained a separate AV2 score for observed harm. At the same
90 edits it found fewer observed failures, but shifted exactly
as many choices to views without an easy reference. It did not
make the fixed position forecast reliable enough to release.
[Conditional harm and support substitution](outputs/publication_readiness_2026_09/av2_observed_harm_v1/README.md).

I separated having an easy reference from passing the harm check
on AV2. The new head predicted reference support better, but
found the same 31 qualifying views at a fixed count of 90. Even
knowing support after the fact left 47 risk failures.
[Support and conditional harm](outputs/publication_readiness_2026_09/av2_projection_support_v1/README.md).

I checked whether the AV2 release score should train only on the
candidate a model would actually choose. At equal training and
intervention counts, the causal and random subsets each found 11
qualifying views, versus 31 for the old cost. The causal subset
also contained fewer easy cases to learn from.
[Selected-path ablation and offline replay](outputs/publication_readiness_2026_09/av2_selected_projection_risk_v1/README.md).

I trained a new AV2 release score on projections from models
that had not seen the source they scored. At the same 90 held
interventions it found 15 qualifying views, versus 31 for the
old cost. None of its five calibrated releases passed.
[Source-excluded release training](outputs/publication_readiness_2026_09/av2_projection_risk_v1/README.md).

I then calibrated when to use the frozen AV2 position projection.
Accounting for easy cases and predicted harm improved the confidence
ranking, but the calibration-selected policy passed only 6 of its
26 held releases. The better reference is not a deployable policy.
[Fixed position release test](outputs/publication_readiness_2026_09/av2_position_release_v1/README.md).

I checked the frozen AV2 position projection on held development
cities. At the same edit budget it qualified on 119 views, up from
89 for the additive reference, with fewer easy cases harmed.
Training on relative trajectories did not improve it, and neither
refinement produced a calibrated release.
[Held reference and relative-loss comparison](outputs/publication_readiness_2026_09/av2_relative_projection_v1/README.md).

I kept the two AV2 recipients and the size of every edit, and
trained only the direction. The learned reference qualified on
105 calibration views, up from 65, and hurt fewer easy cases.
Its relative-trajectory error still rose, so this is a better
reference candidate, not a joint forecasting result.
[Learned fixed-budget directions](outputs/publication_readiness_2026_09/av2_projected_reference_v1/README.md).

I checked whether a better AV2 reference would make the two-edit
comparison more useful. On the calibration cities, 234 views had
a qualifying pair in hindsight, but the learned additive reference
qualified on only 55 of those. A history guard made more pairs
comparable but worsened the reference itself, so I did not keep it.
[Causal reference and exact two-arm capacity](outputs/publication_readiness_2026_09/av2_causal_reference_v1/README.md).

I trained the same AV2 scoring model with and without a
within-scene ranking loss. Neither version beat the additive
cost. The exact audit explains why standalone counts are not
enough: 317 views have a qualifying option, but only 47 offer
a conservative interaction improvement when both arms qualify.
[Within-scene ranking and its matched ceiling](outputs/publication_readiness_2026_09/av2_assignment_order_v2/README.md).

I tested whether separately enforcing support, risk and utility
would make the AV2 assignments more reliable. The hard checks
released eight views and only one passed. The learned checks
released none, and their ranking was worse than the original
cost even without a cutoff.
[Separate assignment constraints](outputs/publication_readiness_2026_09/av2_assignment_constraints_v1/README.md).

I trained assignment costs for the same two AV2 edits. Learning
the separate costs found 89 qualifying views, compared with 63
for the original assignment. Adding the learned interaction term
reduced that to 80. Neither learned policy was reliable enough to
release. Replacing one estimate at a time with the observed result
showed more room to improve gain and risk than the interaction term.
[Learned assignment costs](outputs/publication_readiness_2026_09/av2_assignment_cost_v1/README.md).

I kept the two AV2 displacement profiles but let them move to
different actors. This raised the hindsight ceiling from 176 to
505 views. The past-only rule recovered only 72 and lost accuracy
against the original forecasts, so the added choices are not
evidence that the selector is better.
[Who receives the same two edits](outputs/publication_readiness_2026_09/av2_recipient_reassignment_v1/README.md).

I trained two ways to steer the original AV2 edits while keeping
their recipients and every step's displacement fixed. Neither
improved the original forecast. Even with hindsight and free
directions, most views still cannot qualify; many have no edited
easy reference at those recipients.
[Steering and its exact ceiling](outputs/publication_readiness_2026_09/av2_steered_reference_v1/README.md).

On AV2 I changed the direction of the same two edits without changing
their displacement. Direction and timing found different useful
alternatives, but neither passed release calibration. With both in
the bank, I found a qualifying alternative for almost every already
eligible reference; most original forecasts were not eligible.
[Matched direction and timing](outputs/publication_readiness_2026_09/av2_direction_menu_v1/README.md).

I carried the DroneCrowd history guard over to MovingDroneCrowd++
without fitting on its outcomes. It did not transfer: the independent
forecasts were close to OLS4 on accuracy, but none met the original
risk checks. The guard made no releases.
[MovingDroneCrowd++ transfer test](outputs/publication_readiness_2026_09/moving_history_guard_transfer_v1/README.md).

I then used observed motion to estimate benefit separately from
harm. It improved gain prediction on the same independent forecasts.
Using both checks removed the observed risk failures from the released
set, but retained only eight views from three calibration sources.
[Separate gain and harm checks](outputs/publication_readiness_2026_09/dronecrowd_recording_utility_guard_v1/README.md).

I used the observed track itself to estimate harm. On the same
frozen DroneCrowd forecasts, fitting the first four observations and
checking against the next four predicted recording risk better than
the transferred model. A conservative follow-up selected views from
two sources, where the independent forecasts outperformed the
risk-priced replacements at the same counts.
[Observed-history risk comparison](outputs/publication_readiness_2026_09/dronecrowd_recording_history_risk_v1/README.md).

I tested calibrating both risk ratios over a whole recording rather
than requiring each query to pass on its own. Leaving out one source
at a time, median calibration selected 13 views, of which four passed
both accuracy floors and risk. A larger risk penalty added risk passes
at the same replacement counts, but it lost the accuracy gains.
[Recording-level calibration](outputs/publication_readiness_2026_09/dronecrowd_recording_risk_pool_v1/README.md).

I started two DroneCrowd selectors from the same OLS4 forecast and
changed only the gain target during training. Learning gain directly
against OLS4 hurt at equal replacement counts, even though the same
forecast bank has room for improvement. The query risk head was too
restrictive to select any replacements.
[OLS4-relative gain comparison](outputs/publication_readiness_2026_09/dronecrowd_ols_relative_gain_v1/README.md).

I tried scaling and perturbing the training tracks to bridge the
motion mismatch between two drone datasets. At the same number of
edits, neither version beat OLS4. The noisier training helped the
scene rule meet the risk checks more often, but it still changed
too few paths to compete with OLS4, even in hindsight.
[Matched-count augmentation test](outputs/publication_readiness_2026_09/dronecrowd_observation_augmentation_v1/README.md).

I trained a gain and harm model on a separate moving-drone
dataset and transferred it to DroneCrowd. Gain-only choices
lost to OLS4, and the projected risk budget left too few
edits to catch OLS4 even with perfect forecasts.
[Source-separated full-scene transfer](outputs/publication_readiness_2026_09/dronecrowd_full_scene_transfer_v1/README.md).

On DroneCrowd, keeping the OLS4 slope but moving the forecast
origin to the fitted line improved accuracy in 14 of 15
calibration groups. That did not solve the per-recording harm
checks. A hindsight comparison shows that the same small
forecast bank contains better choices than my past-only rule finds.
[Full-scene motion controls](outputs/publication_readiness_2026_09/dronecrowd_motion_floor_v1/README.md).

I kept the replay-selected forecasts fixed and separately
estimated whether the original path qualifies and whether
the replacement improves on it. This changed calibration
coverage, but did not produce a passing held release.
[Two prerequisites for a replacement](outputs/publication_readiness_2026_09/av2_replay_hurdle_v1/README.md).

I tried changing only the timing of the original two edits.
That kept most of the wider menu's opportunities without
changing whom to edit. Replay retrieved more qualifying
choices than the fitted heads, but could not be calibrated.
[Timing-only substitutions](outputs/publication_readiness_2026_09/av2_incumbent_timing_v1/README.md).

I retrained the timing scorer with a control that saw the
same actions but not the new timing variants. The retrained
model recovered more usable choices on the wider bank, but
still did not beat the original setup after calibration.
[Matched temporal refit](outputs/publication_readiness_2026_09/av2_temporal_refit_v1/README.md).

I kept the edit count and peak displacement fixed and
changed when each edit acts. That gave the joint choice
more ways to beat the original independent path with
hindsight. The existing scorer did worse when given all
the extra options.
[Edit timing and candidate supply](outputs/publication_readiness_2026_09/av2_temporal_menu_v1/README.md).

I trained a selector to predict the gains and risks that
matter against the actual independent choice, rather than
whether an edit looks good on its own. It picked up one more
qualifying held choice than the matched classifier, but
missed more than the earlier ranker and could not be released.
[Matched relative margins](outputs/publication_readiness_2026_09/av2_relative_margins_v2/README.md).

I separated whether an easy reference exists from how much
the edit might hurt it. The estimates improved on supported
scenes. One held edit passed on its own, but the matched
independent choice was still better.
[Easy support and matched release audit](outputs/publication_readiness_2026_09/av2_easy_gate_v1/README.md).

I trained a separate veto on actions chosen by a ranker
that had not seen the city. It improved overall harm
coverage on fixed choices, but did not fix easy-case
harm or make held intervention reliable.
[Out-of-city veto and anonymous replay](outputs/publication_readiness_2026_09/av2_selected_veto_v1/README.md).

I compared the same harm estimate on fixed choices and on
choices it helped select. The estimate covered fewer actual
outcomes after selection, even at the same edit budget.
[Harm coverage before and after selection](outputs/publication_readiness_2026_09/av2_harm_quantile_v1/README.md).

I trained the reference assignment against the same gain
and risk terms used for evaluation. It found more edits with
easy-case support, but the calibrated choices still failed
on held cities. This also uncovered and corrected an
undercount in the earlier hindsight bound.
[Scene-margin assignment and missing-actor correction](outputs/publication_readiness_2026_09/av2_margin_assignment_v2/README.md).

I tested whether the shorter tracks around a scene improve
the intervention decision. They did not fix release. The
same forecast menu does contain better references with
hindsight, so the next question is whether I can learn
to choose them using only observed motion.
[Observed context and reference capacity](outputs/publication_readiness_2026_09/av2_observability_v2/README.md).

I separated the decision to intervene in a scene from the
choice of which agents to edit. This recovered more qualifying
choices, but the one calibration threshold that passed did
not transfer to its held city.
[Paired qualification experiment](outputs/publication_readiness_2026_09/av2_paired_qualification_v1/README.md).

I tried ranking those same choices by what they gain on
observed steps, what they risk on missing steps, and how much
they might hurt easy cases. More edits passed the risk checks,
but the joint result still lost to the fixed additive control.
[Forecast cost and calibration audit](outputs/publication_readiness_2026_09/av2_forecast_cost_v2/README.md).

I tested candidate shapes against the already observed part of
the trajectory. That history signal brought back some choices
that the geometry-only ranker missed. The gain was visible on
complete trajectories, but did not survive the full-grid and
calibration checks.
[Observed-history ranking ablation](outputs/publication_readiness_2026_09/av2_history_rank_v1/README.md).

I retrained the selector on neural forecasts from models that
had not seen the predicted city. More additive edits passed
the risk checks, but a separate score for the interaction
between two edits did not improve the joint result.
[City-excluded selection and interaction ablation](outputs/publication_readiness_2026_09/av2_oof_selection_v1/README.md).

I trained an 8/12 EqMotion control on Argoverse 2 and added its
forecasts to the same edit menu. It increased the number of
usable combinations with hindsight. At the same intervention
budget, the joint decoder did not benefit from those extra choices.
[Public forecast and matched policies](outputs/publication_readiness_2026_09/av2_eqmotion_v1/README.md).

I expanded the Argoverse 2 development sample to 1,024 scenes.
Training on more examples from the same cities helped average accuracy a little
at the same budget, but did not make the risk-controlled edits
reliable in held cities.
[AV2 data-scale experiment](outputs/publication_readiness_2026_09/av2_data_scale_v1/README.md).

I trained a new additive control instead of treating the old one
as fixed. It found more edits with enough easy-case labels to score
risk, but not more edits that passed both risk and accuracy.
Joint re-ranking did not improve it at the same budget.
[Learned additive and joint comparison](outputs/publication_readiness_2026_09/av2_learned_reference_v1/README.md).

I changed the AV2 selector to estimate gain only for qualifying
edits. It found a few useful choices but still could not be
calibrated. The matched comparison also exposed a problem with
the fixed additive control: only two views could support both
arms, versus 57 if either legal forecast could be chosen with
hindsight.
[Supported-gain experiment](outputs/publication_readiness_2026_09/av2_supported_gain_v1/README.md).

I retrained pair selection on Argoverse 2 with a separate calibration
city in each fold. Even with more complete labels, the refit missed
all 47 useful held-city choices. The ranking audit showed why:
the gain estimate filtered out options that the feasibility head
had put near the top.
[AV2 refit and ranking audit](outputs/publication_readiness_2026_09/av2_pair_refit_v1/README.md).

I tried limiting how far a joint forecast could move from the additive
control, at the same intervention budget. That reduced the penalty from
missing labels, but it also removed most of the useful choices and made
the direct forecast comparison worse.
[Same-budget geometry experiment](outputs/publication_readiness_2026_09/opendd_relative_trust_v1/README.md).

I used both feasibility and expected interaction gain to retrieve
eight candidates. It brought back more good choices, but the selected
predictions still did not improve the full-grid result. Most of the
loss came from changed predictions where future pairs were unlabelled.
[Matched shortlist and missing-label attribution](outputs/publication_readiness_2026_09/opendd_joint_retrieval_v1/README.md).

I trained an interaction score specifically for feasible edits.
The shortlist had already removed most of the useful pair choices:
only 42 of 141 calibration opportunities survived the top-eight
filter. The score helped within some of those shortlists, but not
on the full held-location forecast grid.
[Conditional interaction and shortlist audit](outputs/publication_readiness_2026_09/opendd_conditional_pair_gain_v1/README.md).

I trained the selector to put feasible two-agent edits ahead of
similar but infeasible edits. It found more locally valid choices,
but its joint forecasts were worse on average than the matched
additive control. The training target and the interaction metric
were rewarding different decisions.
[Within-window ranking experiment](outputs/publication_readiness_2026_09/opendd_pair_ranking_v1/README.md).

I tried judging a two-agent edit as one decision, rather than releasing
its parts separately. It did not survive calibration on separate OpenDD
locations. There were viable pairs in the menu, but the model rarely
ranked them first.
[Whole-pair admission experiment](outputs/publication_readiness_2026_09/opendd_atomic_pair_admission_v1/README.md).

I checked whether better joint choices still existed after the gate
released its edit slots. Under forced slots, 29 OpenDD recordings
had more than one hindsight improvement that met the constraints.
After the support gate, only one did. The gate leaves too few
two-agent decisions for a better ranker to make a difference.
[Exact matched-slot audit](outputs/publication_readiness_2026_09/opendd_distinct_supported_slates_v1/README.md).

I tested whether predicting the availability of an easy-case reference
would make intervention safer after adding the acceleration path to
OpenDD. It removed some unsupported edits, but also removed most of
the opportunities to make a different joint decision.
[Location-excluded refit and support checks](outputs/publication_readiness_2026_09/opendd_supported_acceleration_v1/README.md).

I added a forecast that lets recently observed acceleration decay over
time. At the same intervention budget it more than doubled the number
of Argoverse 2 scenes with a feasible hindsight win. The frozen selector
used some of the extra capacity, but still did not beat constant velocity
or show a reliable joint gain across all scenes.
[Acceleration and matched replay](outputs/publication_readiness_2026_09/av2_approach_menu_v1/README.md).

I tested whether joint forecast decisions improved on denser Argoverse 2
annotations. At the same edit budget, joint selection still did not beat
the per-agent choice, even when every future step was labelled. The
hindsight comparison shows how rarely the current menu can improve
pair accuracy without compromising other checks.
[Argoverse 2 matched comparison](outputs/publication_readiness_2026_09/av2_native_trajectory_v1/README.md).

On Okutama I separated background motion from trajectory motion using
only the past images. Projecting the forecast back into the moving
camera lowered error at labelled endpoints, but did not establish
an improvement over the full grid.
[Past-camera comparison](outputs/publication_readiness_2026_09/okutama_past_camera_v1/README.md).

I tested the frozen OpenDD policy on a separate aerial dataset,
Okutama-Action. Filtering out generated boxes exposed a coarser
annotation cadence. On those keyframes the joint policy did not
beat constant velocity or establish a conservative pair gain.
[Okutama source audit and transfer](outputs/publication_readiness_2026_09/okutama_label_histories_v1/README.md).

I trained models to recognize edits that can belong to a beneficial,
risk-feasible subset. The broader teacher labels were easier to predict,
but did not improve the decisions at the same edit budget.
[Teacher supervision and matched ranking](outputs/publication_readiness_2026_09/opendd_subset_student_v1/README.md).

I let the model keep the baseline when none of its edits looked useful.
That reduced harm but also left too few paired edits to show a benefit
over choosing agents individually. A hindsight check shows that better
subsets exist among the original edits.
[Abstention and matched release](outputs/publication_readiness_2026_09/opendd_optional_edits_v1/README.md).

I tested whether separate harm predictions and a recording-level risk
ledger could make the same number of edits safer. They did not.
Keeping the original edited recipients preserved the easy reference,
but changing their paths still lost some risk passes.
[Risk ledger and recipient comparison](outputs/publication_readiness_2026_09/opendd_risk_moments_v1/README.md).

I trained a score for the relative-motion gain of a complete replacement,
then tested it at the same edit budget. It improved on my earlier joint
rule, but I could not establish a conservative gain over choosing agents
individually. Most risk failures were already visible in the observed labels.
[Matched comparison and risk breakdown](outputs/publication_readiness_2026_09/opendd_slate_relative_gain_v1/README.md).

I added a turn forecast derived from observed motion and reran the
matched-budget experiment. It raised the number of hindsight wins from
21 to 50. The retrained joint policy produced three qualifying wins over
its own additive control, without a reliable average benefit.
[Turn forecast and matched controls](outputs/publication_readiness_2026_09/opendd_turn_intervention_v1/README.md).

I measured how much relative-motion error could improve with exactly
the same OpenDD edits. Adding a held-source forecast increased the
number of feasible hindsight wins from 9 to 21, but the refitted joint
policy did not produce a corresponding conservative gain.
[Interaction ceiling and learned comparison](outputs/publication_readiness_2026_09/opendd_expanded_interaction_frontier_v1/README.md).

I checked whether PersonPath22 could supply an independent camera test.
Three train/test pairs came from the same source video, and five known
camera codes appeared on both sides. The first camera pilot also lacked
the complete matches needed for a scene comparison.
[Source and camera checks](outputs/publication_readiness_2026_09/personpath22_source_intake_v1/README.md).

I tried the same camera pipeline on a crowded MOT20 recording. Changing
association recovered complete pairs in 10 of 13 windows instead of six.
OLS4 beat constant velocity on the matched tracks, but no window had
labels for the full online roster.
[MOT20 pilot](outputs/publication_readiness_2026_09/mot20_camera_support_v1/README.md).

I checked two ACME recordings at times selected from the camera clock, not
from the annotations. All 24 windows had multiple online tracks, but only
two had a complete labeled pair, and none covered the full online roster.
[Camera coverage study](outputs/publication_readiness_2026_09/acme_camera_roster_v1/README.md).

I put the fixed-budget findings together in two supplement pages: what
the menu permits in hindsight, what the selectors achieve, and what changes
when the same budget moves between windows.
[Supplementary tables and offline replay](outputs/publication_readiness_2026_09/budget_addendum_v1/README.md).

I kept more of the easy reference when moving OpenDD's edit budget. That
made risk easier to estimate, but it did not restore the original policy's
accuracy or pass count.
[Easy-reference guard](outputs/publication_readiness_2026_09/opendd_easy_slot_support_v1/README.md).

I moved the same OpenDD edit slots between neighboring windows. The
joint decoder moved slots in about half the edited pairs, but its risk
and accuracy passes fell from 32 to 26.
[Matched budget timing](outputs/publication_readiness_2026_09/opendd_paired_window_budget_v1/README.md).

I retrained the gain and harm scores on the expanded OpenDD forecast menu.
Joint and additive decisions used the same scores and budget, and each passed
both accuracy and risk in 32 of 606 views.
[Expanded-menu training](outputs/publication_readiness_2026_09/opendd_expanded_costs_v1/README.md).

I added a held-location self-history forecast to both OpenDD selectors at
the same edit budget. The hindsight search found more usable choices, but
the frozen joint selector still passed both accuracy and risk in 34 of 606
views. [Expanded forecast menu](outputs/publication_readiness_2026_09/opendd_self_menu_v1/README.md).

I asked how often an ideal reassignment could beat the independent policy
with exactly the same paths and edit budget. Even with future labels, when
only two-agent choices could change, I found nine wins in 439 estimable
recording-seed views under the original risk and accuracy checks.
[Exact-budget hindsight check](outputs/publication_readiness_2026_09/opendd_exact_slot_headroom_v2/README.md).

I let fine-scale boxes fill gaps in existing VisDrone tracks without starting
new ones. Complete matches rose from 21 to 24, but unmatched histories also
rose from 38 to 75. I did not keep the change.
[Track continuation experiment](outputs/publication_readiness_2026_09/visdrone_continuation_boxes_v1/README.md).

I ran two detector tile sizes on the same VisDrone images. Combining them
gave 65 rather than 50 reference histories with a box at every step, but
still recovered only 21 complete tracks. More boxes alone did not fix tracking.
[Detector scale comparison](outputs/publication_readiness_2026_09/visdrone_detector_tiles_v1/README.md).

I kept the same VisDrone detections and changed only how they were linked.
Compensating for camera motion recovered 21 complete tracks instead of 14.
It still did not give me a window where I could score the whole online group.
[Camera association comparison](outputs/publication_readiness_2026_09/visdrone_association_v1/README.md).

I checked VisDrone-MOT as another source of aerial pedestrian tracks. At
stride four, 191 training windows had at least two complete reference futures.
In the four-window camera check, the tracker recovered 14 of 76 complete
references and no window had labels for the entire online group.
[VisDrone source and camera check](outputs/publication_readiness_2026_09/visdrone_mot_train_intake_v1/README.md).

I checked KIT AIS as a source of aerial tracks. Only one training clip
has enough released frames for the full window, and neither detection
setting that emitted tracks recovered labels for the whole group.
[KIT AIS intake](outputs/publication_readiness_2026_09/kit_ais_source_audit_v1/README.md).

I added the direction of relative motion to the pair score. It helped predict
the pair effect more often, but did not increase the number of recording-seed
views that pass both risk and accuracy checks.
[Directional ablation](outputs/publication_readiness_2026_09/opendd_signed_pair_geometry_v1/README.md).

I tried training the pair score directly on what two edits add beyond
editing each person alone. It recovered a small gain on labeled steps,
but not a reliable gain when missing future labels were accounted for.
[Direct pair-effect experiment](outputs/publication_readiness_2026_09/opendd_direct_pair_effect_v1/README.md).

I gave the joint and independent selectors exactly the same forecast paths,
edit counts and displacement limits on OpenDD. The pair score changed 693
of 1,067 two-edit choices, but did not reliably reduce forecast error.
[Shared-menu comparison](outputs/publication_readiness_2026_09/opendd_slate_shared_menu_v1/README.md).

I used the released HT21 detections to check how often a whole crowd can
be scored. At adjacent frames, most individual histories had complete
futures, but only 36 of 285 multi-agent windows had labels for every
track. At stride 10 it was zero of 22. This matters when comparing a
joint decision to individual choices on the same agents.
[HT21 coverage audit](outputs/publication_readiness_2026_09/ht21_detection_support_v1/README.md).

I checked the same aerial detector on AerialMPT. The source labels
covered 415 complete trajectories in the four development windows,
but the detector and tracker recovered only three. Even allowing
both tile scales and ignoring track identity, just seven had a
detection near every observed position. This points to a detector
coverage problem before the forecast policy can be tested.
[AerialMPT camera experiment](outputs/publication_readiness_2026_09/aerialmpt_camera_prefix_v1/README.md).
Training directly on point annotations recovered 47–56 complete
histories across three seeds. Most emitted tracks still had no
scored future, and none of the joint windows had labels for the
whole online roster. [Point detector and threshold check](outputs/publication_readiness_2026_09/aerialmpt_point_detector_v1/README.md).

I checked whether raw SDD video tracks could support risk learning.
The reference annotations and decoded videos use different pixel rasters.
After aligning them, the best detector recovered 46 complete matches
from 126 online histories. Smoother velocity fits helped on known
outcomes, but the improvement could not be established on all tracks.
[Camera-track study](outputs/publication_readiness_2026_09/sdd_camera_support_v1/README.md).

I tried using weaker detections to fill gaps in those same camera
histories. This gave me 82 complete pedestrian futures instead of 46,
but only two clips had enough matches, both in the same scene.
Among the original unscored tracks, 37 matched cyclists or skaters.
That is a reason to check which agents the labels cover, not just
to lower the detector threshold.
[Confidence and label audit](outputs/publication_readiness_2026_09/sdd_camera_support_v1/confidence_bridge_v2/README.md).

I scored cyclists and skaters separately without changing any
forecast or pedestrian match. That recovered 65 complete futures
on the bridged tracks. But only nine of the 44 windows with at
least two online agents had complete labels for everyone. That
limits what these clips can say about editing forecasts jointly.
[Class and joint-label coverage](outputs/publication_readiness_2026_09/sdd_camera_support_v1/class_support_v2/README.md).

I trained a model to imitate complete editing plans rather than
predicting gain and harm separately. It made some useful choices
on held OpenDD locations, but did not beat independent selection
at the same count and displacement. On the fisheye camera it
intervened more often and still failed the original risk checks.
[Plan imitation study](outputs/publication_readiness_2026_09/source_excluded_slate_v1/README.md).

I tested whether a model could pick the full-size forecasts
that worked in hindsight. It stayed with constant velocity.
Predicting risk directly made it intervene, but those choices
did not meet the original risk rule or beat OLS4. At the same
count and displacement, joint and independent selection picked
the same paths on the development camera.
[Full-path selection experiment](outputs/publication_readiness_2026_09/camera_full_path_selection_v1/README.md).

I trained a small forecaster with noisy observations and tested it
on three development cameras. It improved on the clean-trained
version, but did not consistently beat simple motion forecasts.
On the fisheye pilot, the full-size candidates leave room for a
better selector under the original risk checks. That result uses
hindsight; I have not changed the deployed fallback.
[Camera training study](outputs/publication_readiness_2026_09/camera_noise_control_v1/README.md).

I checked whether better ranking alone could rescue the small camera
interventions. On the development camera, even hindsight cannot make
edits of the original size beat OLS4. A robust velocity fit does beat
OLS4 when it can change the whole forecast, but it exceeds the original
easy-case risk limit. I have not changed the fallback.
[Camera motion study](outputs/publication_readiness_2026_09/wepdtof_camera_capacity_v1/README.md).

I tested the frozen forecasting policy on 13 new WEPDTOF sources.
The strict rule never intervened. When I forced the same number and
size of edits for each method, joint selection did not beat independent
selection, and no recording passed both the original risk check and
the accuracy comparison against both motion baselines. I kept constant
velocity as the fallback. This is an external test of the policy, not
validation of a deployable learned forecast.
[Results and protocol](outputs/publication_readiness_2026_09/wepdtof_external_transfer_v1/README.md) |
[Working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v166/README.md).
I also made a [small offline replay](outputs/publication_readiness_2026_09/main_claims_replay_v1/README.md)
so the headline counts can be checked without downloading videos or models.

I tested whether a new fisheye dataset could support a camera-based
forecasting comparison. On the pilot camera, a fisheye detector
increased stable matches from 22 to 57. On the same 20 targets, the
better motion baseline changed when I used camera histories instead
of edited annotation histories. This pilot informed the protocol
before I scored the other sources.
[WEPDTOF pilot](outputs/publication_readiness_2026_09/wepdtof_source_probe_v1/README.md).

I trained a forecast to stay between constant velocity and OLS4
unless past motion suggested a useful correction. Staying close
to constant velocity passed more risk checks, but it did not
beat OLS4. At the same displacement, learning future label
availability was worse than using the training-set average.
[Reference-bridge study](outputs/publication_readiness_2026_09/moving_reference_bridge_v1/README.md).

I let the same intervention budget move across an entire recording.
With future outcomes known, this added one passing view, from 4 to
5 of 72. In 67 views even arbitrary paths could not catch OLS4 under
that budget. The gap was much larger once missing future positions
were accounted for. I kept the existing fallback.
[Recording-budget study](outputs/publication_readiness_2026_09/moving_record_peak_allocation_v1/README.md).

I trained two versions of the same velocity forecaster to close the gap
to OLS4. The one trained directly against OLS4 was more accurate where
future positions were labeled, but lost once I accounted for missing
labels. Adding recording-risk penalties did not fix that. At the same
intervention times, recipients and displacement limits, neither model
added a passing recording. I kept the current fallback.
[Initial-forecast experiment](outputs/publication_readiness_2026_09/moving_dual_record_forecast_v1/README.md).

I trained the decision head on OpenDD and tested it on a separate
MovingDroneCrowd++ development set. It did not add a passing recording.
Even allowing arbitrary paths in one query at the same displacement
budget could not beat OLS4 in 69 of 72 views. This is a limit of the
current starting forecast and single-query budget, not just the ranker.
[Cross-domain experiment](outputs/publication_readiness_2026_09/cross_domain_pass_transition_v1/README.md).

I tested whether a model could spot a replacement that would erase a
recording's easy reference. Predicting loss for the whole recording
did not beat judging the current query. Breaking the rare event into
two parts helped preserve reference relative to direct prediction,
but still did not improve the overall decision.
[Reference-loss experiment](outputs/publication_readiness_2026_09/opendd_counterfactual_support_v1/README.md).

I let the model change who receives an edit while keeping its time and
displacement fixed. Adding the spline expanded the number of feasible
plans with outcomes known, but training on those plans did not beat the
past-prefix rule. Fixing the learned recipients lost much of that
potential even when I re-optimized path shape afterwards.
[Recipient study](outputs/publication_readiness_2026_09/opendd_feasible_recipient_imitation_v1/README.md).

I recalibrated the spline choice on the slots the controller actually
uses. The guarded rule made no spline substitutions and did not pass
more recordings. Only 55 of 1,402 edited queries had two slots, which
helps explain why coupling the last shape choice changed so little.
[Streaming calibration study](outputs/publication_readiness_2026_09/opendd_stream_spline_calibration_v1/README.md).

I tested whether a learned rule could choose the better path at the same
OpenDD intervention locations. The risk-aware rule chose the spline only
four times, and choosing jointly gave the same result as choosing each
track independently. It did not beat the simpler rule based on past
motion. I am checking whether the examples used for calibration match
the intervention decisions the controller actually makes.
[Nested selection experiment](outputs/publication_readiness_2026_09/opendd_nested_spline_choice_v2/README.md).

I reran the velocity forecast on the better-labeled OpenDD roster,
matching the training budget of an EqMotion control. It improved
on constant velocity on average, but not reliably on every location.
The new forecasts expanded what was possible with outcomes known;
simply swapping them into the old intervention slots did not help.
[Matched OpenDD study](outputs/publication_readiness_2026_09/opendd_matched_spline_v1/README.md).

I trained a forecaster that integrates a learned velocity curve.
One version improves on OLS4 where future positions are labeled,
but it does not beat OLS4 after accounting for missing positions.
Its paths also do not add a verified plan at the same edit counts.
Returning gradually to OLS4 narrows the gap without clearing it.
[Velocity forecast experiment](outputs/publication_readiness_2026_09/moving_causal_velocity_spline_v1/README.md).

I added nine OLS-based paths to see whether a better motion forecast
could help at the same edit counts. The new paths improved several
plans in hindsight but did not add a recording that passed the
original rule. In 28 of 36 excluded views, even allowing the same
edits to move anywhere in the recording could not beat OLS4 in the
best case. I am working on the forecast options before another ranker.
[Path-bank experiment](outputs/publication_readiness_2026_09/moving_ols_anchor_menu_v2/README.md).

I tried to recover the joint policy's lost accuracy without increasing
its predicted risk or editing more tracks. Neither learned gain model
improved the conservative score, and the worst easy cases got worse.
Even with future outcomes available, fractional choices could recover
at most 0.058 percentage points against OLS4 on average. I kept the
fallback and will revisit the paths and allocation constraints.
[Fixed-budget experiment](outputs/publication_readiness_2026_09/moving_cv_fixed_budget_gain_v1/README.md).

I separated predicted harm, missingness and easy reference
support. That did not improve independent choices at the same
counts. At lower, matched counts, joint selection passed more
risk checks than independent selection, but gave up forecast
accuracy. The next question is whether I can recover that
accuracy without giving back the risk improvement.
[Risk-component experiment](outputs/publication_readiness_2026_09/moving_cv_risk_components_v1/README.md).

I tried keeping a running estimate of the risk already taken in
each recording. The state helped predict accuracy gains but did not
improve the risk decision. Both the learned model and a direct
arithmetic control lost at the same edit counts. I am focusing next
on estimating easy-case harm and the support for that estimate.
[Recording-prefix experiment](outputs/publication_readiness_2026_09/moving_cv_prefix_state_v1/README.md).

I trained two models on edits that differ by one action and cross
the recording's risk limit. Teaching the risk preference reduced
the worst easy-case change relative to teaching accuracy alone,
but neither passed more than one of 72 development checks at
the same edit counts. I need the next model to account for the
rest of the recording's plan when it judges an edit.
[Matched risk-boundary experiment](outputs/publication_readiness_2026_09/moving_cv_risk_crossing_v1/README.md).

I expanded the set of scene decisions by asking a solver to
combine paths under different predicted risk budgets. With outcomes
available, the wider set finds 11 valid views instead of eight.
The same path bank can support at least 27 if choices are made
per agent, so many useful combinations are still missing. The
frozen model did not benefit from the extra options at matched
counts. I am using this to guide what the next model learns.
[Risk-budgeted options](outputs/publication_readiness_2026_09/moving_cv_risk_frontier_v1/README.md).

I let the model decide how many forecasts to replace as well as
which paths to use. It gained accuracy on labeled steps relative
to keeping the old counts, but did not beat independent selection
at the same edit counts. With outcomes available, allowing the
number of edits to vary recovered eight valid development views
where the old counts recovered none. The next problem is generating
more options that meet the same risk rule.
[Variable-count experiment](outputs/publication_readiness_2026_09/moving_cv_joint_count_v1/README.md).

I trained the risk model on the paths the forecaster actually
proposes. Predicting a higher harm quantile reduced underestimation,
but the worst easy cases got worse. I also checked what a perfect
ranker could do: keeping the same number of edits, even the full
path bank could not beat OLS4 in 45 of 72 development views. The
next change is to learn how many forecasts to replace as well as
which paths to use.
[Selected-edit risk study](outputs/publication_readiness_2026_09/moving_cv_selected_harm_v1/README.md).

I tested whether the model could learn how far to move a forecast
instead of inheriting a fixed displacement limit. It predicted gain
more accurately, but allocating the same total movement unevenly
did not improve the conservative result. I have not changed the
fallback policy. For DroneCrowd, I use 15 source groups for
calibration and keep 30 separate groups for independent confirmation.
[Amplitude experiment](outputs/publication_readiness_2026_09/moving_cv_amplitude_response_v1/README.md).

I let the model swap in new trajectories while keeping edit counts
and displacement fixed. A simple geometric rule reduced the worst
easy-case change but still failed calibration. Even allowing arbitrary
paths and pooling the same displacement across a recording could not
beat OLS4 in 67 of 72 development views. Changing who gets edited is
not enough at that displacement budget.
[Recipient exchanges and displacement limits](outputs/publication_readiness_2026_09/moving_cv_matched_recipient_exchange_v1/README.md).

I held the earlier model's choice of trajectories fixed and compared
learned direction changes with simply shrinking the old paths.
The learned changes did not improve the conservative score.
Even arbitrary directions within the original radii could not
beat the OLS4 reference in 67 of 72 development views. I need to
improve which trajectories receive a correction, not only its shape.
[Fixed-recipient experiment and upper bound](outputs/publication_readiness_2026_09/moving_cv_fixed_recipient_support_v1/README.md).

I tested choosing who to edit separately from choosing a path.
Learning from many feasible plans did not help the first decision.
The path ranker reduced the worst easy-case error, but shrinking
the original paths explained most of the observed accuracy change.
At the same displacement, the new path still lost on the
conservative comparison. I kept the existing fallback.
[Recipient and path experiment](outputs/publication_readiness_2026_09/moving_cv_recipient_path_v1/README.md).

I checked how many edits the full bank actually needs. At a quarter
of a recording's rows, it can still beat both references under the
original rule in 38 of 72 hindsight views. Teaching a model from
smaller feasible plans did not reproduce that gain. Much of the
failure came from choosing the wrong trajectories to edit.
[Edit budgets and teacher experiment](outputs/publication_readiness_2026_09/moving_cv_minimum_budget_v1/README.md).

I separated the candidate menu from the number of edits it could
make. Knowing the outcomes, the narrow menu could find valid plans
in 5 of 72 views. The full bank recovered 19 at the same query
counts and 53 when counts were free. This identifies room for a
better forecast decision, but it does not show the model can find
those plans in advance.
[Candidate scope and edit counts](outputs/publication_readiness_2026_09/moving_cv_budget_menu_factorial_v1/README.md).

I trained a forecaster to rank whole groups of edits using examples
that had passed the original recording-level rule. That supervision
did not improve the conservative comparison at equal edit counts.
A hindsight check found that the fixed query budgets and small
menu left valid choices in only 5 of 72 views. I am keeping the
existing fallback while I work on what the forecaster can choose.
[Whole-query supervision and capacity](outputs/publication_readiness_2026_09/moving_cv_slate_supervision_v1/README.md).

I separated learning when annotations are missing from learning
which edits are worth making. Step-specific availability prediction
improved its own score, but not the conservative forecast comparison
at equal intervention counts. Penalizing underestimated easy harm
helped more recordings pass while worsening the single worst one.
[Missingness and easy-risk ablation](outputs/publication_readiness_2026_09/moving_cv_lead_risk_v1/README.md).

I retrained the risk estimator to recognize which edits actually have
easy reference support. It recovered far more of those opportunities,
but the wider policy still missed the original risk requirement.
At equal intervention counts, a new gain ranker helps on observed
steps; the gain does not survive the missing-step bound.
[Signed risk and matched gain](outputs/publication_readiness_2026_09/moving_cv_signed_coverage_v1/README.md).

I tried sharing predicted risk slack across agents so the forecaster
could intervene more often. The policy barely expanded coverage and
still did not qualify. With outcomes available, the same full menu
could beat both references under the original rule in 53 of 72
views. The next question is how to identify those combinations
before seeing the result.
[Coverage and support diagnosis](outputs/publication_readiness_2026_09/moving_cv_pooled_coverage_v1/README.md).

I trained two trajectory mixtures to see whether changing the path itself
could recover the missing gains. Making the weights vary along the forecast
did not improve the shared-weight control. More importantly, even perfect
edits at the existing budget could not catch both motion references in 64
of 72 views. This points to a coverage problem before another shape change.
[Proposal and coverage study](outputs/publication_readiness_2026_09/moving_cv_lead_mixture_v1/README.md).

I tested whether it helps to spend the same number of edits at different
points in a recording. Neither the retrospective reallocation nor the
streaming policy improved the full risk result. Even with outcomes known,
the fixed proposal menu could beat both motion references under the
original risk rule in only three of 72 views when the timing was freed.
[Recording allocation study](outputs/publication_readiness_2026_09/moving_cv_record_allocation_v1/README.md).

I compared learning a preference between two paths with predicting each
path's value on its own. At the same edit count, the preference loss reduced
the worst known easy regression from 32.11% to 0.84%, but did not improve
the number of recordings that met the full risk rule or show a reliable
utility gain. The calibration still rejected both policies.
[Matched ranking experiment](outputs/publication_readiness_2026_09/moving_cv_paired_nomination_v1/README.md).

I tested whether showing the risk model more candidate paths would recover
some of the missed gains. Giving it the entire menu, or just keeping the OLS4
alternative, reduced the worst easy-case regression at the same edit count.
Fewer recordings met the full risk rule, though, and no setting qualified on
calibration. I kept the existing fallback.
[Screening-order experiment](outputs/publication_readiness_2026_09/moving_cv_screen_order_v1/README.md).

I trained a model to screen edits for predicted harm and missing support.
On held fitting sources, screening increased risk passes at the same edit
count, but no calibration qualified. A separate hindsight check showed why
the shortlist matters: retaining the OLS4 path as an option recovered
combinations that withholding edits from the initial ranking could not.
[Optional editing and shortlist diagnosis](outputs/publication_readiness_2026_09/moving_cv_admission_v1/README.md).

I trained a new selector on forecasts from source-excluded fitting models.
At the same number of edits, the learned pair term improved pair error on
observed steps, but not the conservative full-trajectory score. Matching
displacement did not change that result. I did not use it to replace CV.
[Learned pair choice](outputs/publication_readiness_2026_09/moving_cv_joint_selector_v1/README.md).

I gave the same hindsight solver access to frozen neural forecasts as well
as the motion baselines. At the same cap of eight edits per query it found
new feasible choices on five development recordings. Most of those choices
could not be reproduced by the analytic paths even at the exact same edits
and displacement, but the selected forecasts still had worse pair geometry
than OLS4. This is evidence about the menu, not a deployable selector.
[Frozen neural choices](outputs/publication_readiness_2026_09/moving_cv_joint_feasibility_v1/README.md).

I also tested the other starting point: keep CV on every untouched track.
One edit per query is enough to improve CV in hindsight, but not enough to
catch OLS4 on 30 of 34 recordings even with idealized edits. The five tested
motion alternatives have no positive ceiling over OLS4 at that budget under
the relaxed risk constraints.
[Edit-budget comparison](outputs/publication_readiness_2026_09/moving_cv_anchor_budget_v1/README.md) |
[中文复现](reproducibility/M3W_CURRENT_ZH.md).

I tested whether changing just one OLS4 forecast per query could meet my
original risk rule. Even if I knew which track to fix, the harm left on the
untouched tracks would still exceed the limit on all 34 development
recordings. I keep OLS4 as an accuracy comparator; this sparse OLS4 policy
is not a deployment candidate.
[Matched edits and risk bound](outputs/publication_readiness_2026_09/moving_motion_cost_stability_v1/README.md).

I compared predicting the winning motion forecast with predicting how much
it would help or hurt. With the same tracks and edit sizes, the cost model
guessed the winner less often but made less costly mistakes on labeled
steps. Splitting the cost into early and late parts did not help. Missing
labels still prevent me from claiming a gain across the whole forecast.
[Cost versus winner experiment](outputs/publication_readiness_2026_09/moving_motion_choice_cost_v1/README.md).

I tested whether a short backtest inside the observed trajectory could
tell a model which motion forecast to trust. It improved some observed
errors, but the advantage nearly vanished at the same edit size. The
backtest still ranked most candidate pairs correctly at step 12, yet
its chosen forecast lost to OLS4. I did not use it to replace the baseline.
[Past-backtest experiment](outputs/publication_readiness_2026_09/moving_prefix_expert_mixture_v1/README.md).

On DUT I checked the apparent EqMotion advantage against a stronger
four-point motion baseline. OLS4 had lower error in both sites and all
three seeds, including on the full earlier roster. At the same edited
objects and displacement, the neural direction still did not beat
OLS4, even under the favorable bound for missing labels.
[DUT matched-motion comparison](outputs/publication_readiness_2026_09/dut_matched_motion_v1/README.md).

On OpenDD I tested whether penalizing missing-step disagreement during
training produces a better forecast. Penalizing disagreement with
constant velocity gave a small, uncertain gain against it. The two-reference
loss still lost to ordinary masked training at the same per-agent edit
size, and none of the new forecasts met my criteria to advance.
[OpenDD completion-loss comparison](outputs/publication_readiness_2026_09/opendd_completion_forecast_v1/README.md).

On Hannover I checked whether spending the same total trajectory
adjustment on different pedestrians could catch OLS4. It could not:
even twice the original budget fell short in an optimistic calculation
with arbitrary forecast directions. This was a diagnostic on previously
examined recordings, not a new independent evaluation.
[Hannover shared-budget audit](outputs/publication_readiness_2026_09/hannover_shared_radius_v1/README.md).

I tested whether neighbors observed at the same time improve
trajectory choice on TUMDOT. Even after matching how often the
models changed the past-motion prediction, the model with correctly
aligned neighbors lost to both the no-neighbor and shuffled controls.
[TUMDOT neighbor comparison](outputs/publication_readiness_2026_09/tumdot_peer_choice_v1/README.md).

I checked whether the TUMDOT forecast menu itself was limiting the
selector. With outcomes available, all 27 half-radius views have a
solution that beats three references and meets the risk rule.
Training on those joint decisions still lost to training on each
object separately. Most of the previous model's disagreement debit
came from partially labeled tracks, not completely unlabeled tracks.
[Three-reference audit and teacher comparison](outputs/publication_readiness_2026_09/tumdot_joint_teacher_v1/README.md).

On TUMDOT, I kept the edited objects and displacement fixed and learned
when to replace a past-motion forecast. The guarded model passed on
more individual recordings, but its average conservative performance
still lost to both constant velocity and the fixed past-motion rule.
[TUMDOT matched selection](outputs/publication_readiness_2026_09/tumdot_incumbent_selector_v1/README.md).

I fixed the first-edit opportunity and displacement, then compared
learning to rank a whole query with scoring each candidate separately.
Diversity helped recover some qualifying options in a shortlist.
It did not make the top choice better, and the listwise
model lost to its pointwise control.
[First-offer ranking](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/opening_slate_v2/README.md).

I tested whether waiting to make the first edit helps when its size is
fixed. At the same count and displacement, the learned timing rule
did not beat acting immediately. A second opportunity did not increase
the number of recordings with a qualifying edit; the ranker still
missed most of the qualifying actions already available.
[First-edit timing](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/radius_opening_v1/README.md).

I tested whether a small change in who gets edited can restore easy-case
risk support without adding edits. With outcomes available for diagnosis,
it restored support in some recordings. It never restored support and
improved on all references in a view where OLS4 was demonstrably stronger
than constant velocity. In 12 undefined views the original policy made
no edits, so there was nothing to exchange.
[Small support exchange](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/small_support_exchange_v1/README.md).

I traced the pedestrian policy's loss to disagreement on tracks with no
future labels, especially when it replaced the original recipients. Keeping
those recipients and learning only which direction to change reduced the
loss, but still did not improve the original rule.
[Fixed-recipient experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/incumbent_direction_v2/README.md).

I tested whether local crowd geometry could predict which future labels
would be missing. It improved that prediction and steered edits away
from unlabeled tracks, but the edited trajectories still lost to the
previous rule at the same budget. I kept the original policy.
[Pedestrian observability experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/observability_v1/README.md).

I separated a larger forecast bank from a larger intervention budget.
More edits opened more feasible plans against OLS4 than seven extra
forecast variants did. The learned risk screen did not recover that
capacity: its observed gains were offset by missing-label uncertainty.
[Pedestrian budget experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/budget_risk_v1/README.md).

I tested one sparse edit per recording on the pedestrian development
sources. The learned rank found more qualifying edits than the old
gain rank at the same budget, but it did not close the gap to OLS4.
[Sparse pedestrian intervention](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/sparse_support_v1/README.md).

I trained the support model to distinguish a new recording pass from
the loss of an existing one. It rescued more recordings than the
previous scoring rule but also broke more passes. Most opportunities
were concentrated at one location, where the model recovered none.
[Pass-transition experiment](outputs/publication_readiness_2026_09/opendd_pass_transition_v1/README.md).

I gave the support model more forecast directions without increasing
the size of an edit. More useful actions existed, but the learned rule
rarely selected them. The missed-opportunity audit found that the
main problem was knowing when to act.
[Matched support directions](outputs/publication_readiness_2026_09/opendd_support_directions_v1/README.md).

I trained a separate guard to predict when an early edit would erase
the only easy reference. The guard reduced reference loss but did
not beat the original policy. Even with hindsight, choosing from the
same early proposals left little room to improve.
[Reference-loss guard](outputs/publication_readiness_2026_09/opendd_reference_loss_v1/README.md).

I kept the gain edit fixed and tested whether the model could identify
support earlier. A local reference head never intervened. Training
on whole-recording outcomes made it intervene, but some early edits
replaced later ones that supplied the only easy reference.
[Early support experiments](outputs/publication_readiness_2026_09/opendd_early_reference_v1/README.md).

I trained a model to decide whether to make a gain edit now or wait for
a later offer, without changing the support edit. It lost to the
same-budget immediate control. In most expired waits, the original
support edit never arrived or arrived too late.
[Gain timing experiment](outputs/publication_readiness_2026_09/opendd_gain_wait_v1/README.md).

I kept the support edit fixed and trained a model to change only the
gain edit. It lost one previously defined easy reference instead of
20 when the model could change the gain recipient at any time, but
still did not beat the original policy. An analysis using future
outcomes shows more room if the gain edit can wait.
[Protected gain experiment](outputs/publication_readiness_2026_09/opendd_protected_gain_v1/README.md).
For DroneCrowd I use 15 source groups for calibration and reserve 30
for independent confirmation. The confirmation groups remain sealed.

I held the intervention budget fixed and learned which tracks to
edit. Relative-path features helped against a retrained ablation,
but ranking did not beat the original policy. Changing recipients
explained the loss of easy-reference coverage.
[Matched-budget ranking](outputs/publication_readiness_2026_09/opendd_budget_rank_v1/README.md).

I let both edit slots move and trained a model to decide when to
use them. It made fewer edits and lost to the earliest-use control
at the same budget. An explicit deadline recovered coverage but
did not recover an advantage over that control.
[Two-role timing experiment](outputs/publication_readiness_2026_09/opendd_dual_clock_v1/README.md).

I compared learning forecast directions against one motion baseline
and against two. The two-baseline head did better than the CV-only
head but did not improve on the original neural edits. Fixing the
edited tracks also left the same recordings without an easy reference.
[Matched direction experiment](outputs/publication_readiness_2026_09/opendd_dual_direction_v1/README.md).

I tested whether a support edit should precede the gain edit.
It did not help at the same budget. A separate analysis shows that
changing both edit times creates much more room, and allowing
simple motion directions adds options without larger displacements.
[Schedule and forecast capacity](outputs/publication_readiness_2026_09/opendd_support_schedule_v1/README.md).

I trained the second-edit selector on the outcome of both edits
over a full recording. It reduced error slightly over easy
ranking but did not beat the matched local-label control reliably,
and fewer recordings passed. I did not adopt it.
[Whole-recording supervision](outputs/publication_readiness_2026_09/opendd_record_support_v1/README.md).

I checked what happens to risk when a second edit is added. It made
some previously unscorable recordings pass and pushed others over the
limit. More edits do not necessarily make the risk go up or down.
[Recording risk transitions](outputs/publication_readiness_2026_09/opendd_support_monotonicity_v1/README.md).

I trained a model to choose which track to edit without changing the
number, timing, or size of the edits. Ranking for forecast gain
reduced error but fewer recordings passed the joint rule. Adding a
guard restored the pass count without a reliable error gain.
[Fixed-query recipient experiment](outputs/publication_readiness_2026_09/opendd_support_admission_v1/README.md).

I tested whether holding a support edit for a later query helps. It did
not reliably improve forecast error over spending the same budget
immediately in those recordings. An offline analysis found room to improve both the timing
and the choice of which track to edit.
[Support timing experiment](outputs/publication_readiness_2026_09/opendd_support_stop_v1/README.md).

I separated the gain edit from a second, smaller edit intended to
recover reference support. At the same second-slot times, easy
ranking passed more recordings than gain ranking but reduced
forecast utility. Neither meets the full recording-level rule.
[Two-slot comparison](outputs/publication_readiness_2026_09/opendd_two_reserves_v1/README.md).

I removed the original edits and kept at most one low-amplitude
change per recording. It improves aggregate error over constant
velocity on OpenDD, but still lacks enough recording-level support
to qualify as a deployment policy.
[Fixed shapes and sparse edits](outputs/publication_readiness_2026_09/opendd_reserve_shape_v1/README.md).

I gave each recording one optional extra edit to see if the missing
reference support could be recovered online. The simple easy rule
increased the number of qualifying recordings but reduced forecast
utility. Predicting gain improved utility over the incumbent, not CV.
[One-edit reserve experiment](outputs/publication_readiness_2026_09/opendd_reserve_v1/README.md).

I tested whether learning from feasible whole-recording assignments
helps at the same edit count. It improved some risk counts but lost
forecast utility. A separate one-edit analysis found that missing
support can sometimes be repaired without changing the risk rule.
[Recording-level supervision and support](outputs/publication_readiness_2026_09/opendd_record_teacher_v1/README.md).

I trained a pairwise ranker to choose recipient sets without changing
when edits happen or how many are used. It improved over the fixed
ranking on OpenDD, but not reliably over a learned regression control
or constant velocity. The recording-level risk problem remains open.
[Same-clock ranking results](outputs/publication_readiness_2026_09/opendd_slate_rank_v1/README.md).

I compared adaptive recipient sets with a fixed ranking at the same
number of edits in each query. The adaptive sets widen the hindsight
opportunity, but the learned rule does not improve forecast error at
that matched clock. This keeps the focus on learning which edits help.
[Joint recipient and timing experiment](outputs/publication_readiness_2026_09/opendd_slate_queue_v1/README.md).

I tested whether learning to save an edit for a later query improves
on scoring the current query alone. It did not reliably help. At the
same budgets, fixing who gets edited first also cut the two-query
hindsight ceiling from 223 to 84 of 606 views.
[Learned timing and recipient limits](outputs/publication_readiness_2026_09/opendd_credit_value_v2/README.md).

In an earlier experiment I asked whether the same neural edits would work better at different
times. With the total fixed in each recording, letting edits move up to
two queries later raises the hindsight ceiling from 88 to 223 of 606
views. None of the past-only waiting rules reliably improves forecast
utility at the same number of used credits.
[Timing experiment](outputs/publication_readiness_2026_09/opendd_release_credit_v1/README.md).

I trained a selector on the specific neural forecast's gain and harm,
using forecasts produced without either the fitting row's site or the
final evaluation site. At the same query budgets it did not reliably
beat the difficulty rule or constant velocity. Even with outcomes
known, the original risk rule limits this schedule to 88 of 606 views.
[Candidate-specific experiment](outputs/publication_readiness_2026_09/opendd_candidate_advantage_v1/README.md).

I separated the neural correction's gains on difficult tracks from its
losses on easy tracks. The losses canceled the gains. A learned predictor
of CV difficulty did not reliably beat a simple motion rule at the same
switch counts, and it did not improve the fixed-budget policy.
[Forecast error breakdown](outputs/publication_readiness_2026_09/opendd_component_guard_v1/README.md) |
[Difficulty gate](outputs/publication_readiness_2026_09/opendd_hardness_gate_v1/README.md).

I compared a compact EqMotion adaptation with two strong motion baselines
on held-out OpenDD development locations. The ego-only model in a moving
frame beat OLS4, but did not beat constant velocity. Adding the nearest
neighbors hurt this version. The neural trajectories also did not improve
on the donor-prefix control at the same budget.
[Trajectory comparison](outputs/publication_readiness_2026_09/opendd_comoving_eqmotion_v1/README.md).

I also tested whether older forecasts, whose outcomes were already
observable, could help choose who receives an edit. At the same edit
times and radii, using the same pedestrian's older errors did not
improve the recording-level result. The frozen recipients remain better.
[Earlier-error experiment](outputs/publication_readiness_2026_09/opendd_feedback_recipient_v1/README.md).

I trained a model to choose between the two trajectory shapes without moving
any intervention. It did not reliably improve on the past-only rule. Even
with hindsight, the same intervention positions could qualify in at most
157 of 606 recording views; 334 lacked the easy-reference support required
by the risk test. [Shape selection and its limit](outputs/publication_readiness_2026_09/donor_shape_choice_v1/README.md).

I tested whether past motion from another OpenDD site could improve the same
trajectory edit. With the recipients, timing and displacement fixed, the
new shapes increased the number of recording views that passed both risk and
forecast checks from 77 to 90 out of 606. The overall conservative forecast
score did not improve reliably, so I kept the original policy.
[Matched trajectory experiment](outputs/publication_readiness_2026_09/donor_motion_shape_v1/README.md).

I also tested whether training against both CV and OLS4 improves forecasts
when the intervention budget and recipients are held fixed. The small gain on
observed labels did not survive the conservative missing-label comparison or
the safety checks. [Matched training experiment](outputs/publication_readiness_2026_09/native_dual_reference_forecast_v1/README.md).
I then penalized error above the better reference on each training example.
That changed the learning signal but still did not improve the conservative
score or the safety checks. [Pointwise harm experiment](outputs/publication_readiness_2026_09/native_pointwise_reference_forecast_v1/README.md).

For missing trajectory labels, I also tested how well two extrapolation
centers cover positions I deliberately hid from released tracks. Anchored
OLS4 covered more complete 12-step tails than a fixed-position center.
The ranking of joint and additive selection can still reverse under coherent
completions around the new center.
[Missing-label stress test](outputs/publication_readiness_2026_09/native_completion_occlusion_v1/README.md) |
[Motion-centered rerun](outputs/publication_readiness_2026_09/native_motion_centered_completion_v1/README.md).

On OpenDD, I also required the three training seeds to share one
unknown position at every missing target step. This tightened the
error bounds, but the apparent advantage of time-aligned motor
context over the pedestrian-only and shuffled controls could still
reverse. [Shared-label comparison](outputs/publication_readiness_2026_09/opendd_seed_shared_completion_v1/README.md).

## DroneCrowd Calibration

I assigned DroneCrowd by source group: 15 groups (40 recordings) for
threshold calibration and 30 groups (72 recordings) for independent
confirmation. I tested the frozen policy on calibration only. None of the
four thresholds qualified under the original 2% rule, so I kept the
confirmation groups sealed. At the same switch
count, ungated joint selection passed 69 of 120 recording-seed views versus
18 for independent selection, but only two more than pair-additive selection.
OLS4 was much stronger on trajectory error (ADE 1.468 vs. 2.984 image
pixels for ungated joint selection). I do not deploy the new policy.
[Calibration results](outputs/publication_readiness_2026_09/dronecrowd_calibration_readout_v1/readout/report.md) |
[Working manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v24/README.md).

I then checked whether the original two-switch budget could reach OLS4
even with perfect predictions on the best recipients. Its upper bound
was +5.29 points versus CV, well below OLS4's +51.29. Without that cap,
the frozen candidates contain a zero-harm selection, but finding it
requires knowing the future. [Capacity experiment](outputs/publication_readiness_2026_09/dronecrowd_perfect_switch_ceiling_v1/README.md) |
[Updated paper](outputs/publication_readiness_2026_09/evidence_manuscript_v37/README.md).

I then let the model choose how far to move each prediction from CV
toward OLS4. The learned gain rule still trailed OLS4. The initial
risk search chose no changes; after I repaired the search, the policy
passed 99 of 120 recording-seed checks. Assigning exactly the same
blend strengths using a simple past-motion backtest passed 114 and
gave slightly better forecasts. I kept the original 2% rule, did not
select a deployment policy, and left the confirmation groups sealed.
[Experiment and matched comparison](outputs/publication_readiness_2026_09/dense_motion_calibration_v1/README.md) |
[Updated manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v38/README.md).

I also tested whether a past-motion backtest can choose whom to adjust
without relying on the learned policy. It passed all 40 opened
DroneCrowd recordings, compared with 35 when I held the number and
strength of adjustments fixed but ranked people using only the latest
backtest. The error difference between those two rules was uncertain.
The same standalone rule failed on the original fitting sources, and
full OLS4 still gave lower error on DroneCrowd. I have not deployed it.
[Standalone comparison](outputs/publication_readiness_2026_09/standalone_hindcast_v1/README.md) |
[Updated manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v39/README.md).

I tested whether removing shared motion from the velocity correction
would explain that reversal. Holding both the recipients and correction
lengths fixed, the new direction made forecasts worse on the opened
DroneCrowd clips and did not fix fitting-source risk. The gap remained
even among queries with similar levels of shared motion.
[Same-norm comparison](outputs/publication_readiness_2026_09/shared_correction_audit_v1/README.md) |
[Updated manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v40/README.md).

I then held each forecast and its target labels at the same time
indices, changing only the past points used to estimate velocity.
Wider histories helped on the opened DroneCrowd clips but hurt the
conservative score on the fitting sources. Even with hindsight and
arbitrary mixtures of the four fits, none of the 24 fitting recordings
could pass at the same recipients and half-blend amplitudes.
[Sampling and capacity experiment](outputs/publication_readiness_2026_09/past_sampling_audit_v1/README.md) |
[Updated manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v41/README.md).

I tested whether the earlier failure was due to the paths or to who
received them. With hindsight, changing recipients at the same
per-window counts worked on 8 of 24 fitting recordings. Redistributing
the same recording totals raised that to 14. With freely chosen
counts, full-strength straight paths were sufficient to reach 18;
curved paths added no passing recordings. My leave-source-out
predictor still passed none, so this is not a deployment result.
[Allocation and learned-risk comparison](outputs/publication_readiness_2026_09/recipient_release_capacity_v1/README.md) |
[Updated manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v42/README.md).

I trained paired risk models at the same recording totals to check
whether explicit easy-reference supervision helps. It made reference
overestimation less frequent, but neither model passed the risk
check. Holding their recipients fixed and choosing the best paths
with hindsight still did not pass. The result points to the choice
of recipients rather than another path swap.
[Matched training and diagnosis](outputs/publication_readiness_2026_09/matched_support_moments_v1/README.md) |
[Updated manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v43/README.md).

I trained a second pair of models to estimate whether each person
had any feasible replacement, keeping path nomination fixed.
The accuracy contrast to the prior recipients was uncertain and
none of the matched-budget arms passed. The predicted easy-risk
constraint already excluded the original counts, so another search
grid was not the answer. A query-local fallback rarely intervened
and added no forecast gain.
[Recipient feasibility](outputs/publication_readiness_2026_09/recipient_feasibility_v1/README.md) |
[Updated manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v44/README.md).

I then separated fitting, risk calibration, and scoring by source group.
The new ranking did not beat the gain-only ranking at the same number of
changes, and neither found a threshold that passed the 2% rule. Four of
nine calibration splits had a recording with no easy examples; the other
five had risk-feasible actions in hindsight, but the scores could not
find them. I have kept the DroneCrowd confirmation groups sealed.
[Source-disjoint experiment](outputs/publication_readiness_2026_09/nested_recipient_calibration_v1/README.md).

I also tested whether changing the timing of an edit helps when the
recipient, direction, and maximum offset stay fixed. The learned
timing did not beat uniform scaling. Even with future labels used to
choose the best timing, only four of 24 development recordings could
pass the original risk rule. That makes changing the recipients or
candidate paths a better next test than tuning the timing again.
[Matched timing experiment](outputs/publication_readiness_2026_09/query_temporal_envelope_v1/README.md).

I then let the model change the direction at each step, keeping the
size of every edit fixed. Adding local neighbor history did not help.
An outcome-aware search can do more if it also changes which agents
receive edits: at the same query counts, it passed the risk rule and
beat both references in ten of 24 recordings. That is room to improve
the allocation, not evidence that the learned model can do it.
[Direction and allocation study](outputs/publication_readiness_2026_09/matched_direction_v2/README.md).

I also let the policy leave a query unchanged or edit at most two
trajectories. At the same number of edits, joint choice passed more
recording-level risk checks than a simple risk ranking, but lost
conservative forecast utility. Even with hindsight and no risk
restriction, that two-person action set could not beat OLS4 in 19 of
24 development recordings. The operating policy is unchanged.
[Optional allocation study](outputs/publication_readiness_2026_09/optional_allocation_v2/README.md).

I replaced the quarter-strength edits with a full-strength velocity
forecast that learns how to weight the past observations. It beat a
fixed robust fit on the development roster, but did not establish a
gain over OLS4 or pass the risk rule. The two references already
disagree on trajectories without future labels, so shrinking the
model's deviations cannot remove most of the conservative debit.
[Weighted velocity and reference disagreement](outputs/publication_readiness_2026_09/weighted_velocity_v1/README.md).

I checked the missing moving-drone labels against the publisher's
release. The converter had not dropped any available observations;
most of the disagreement cost comes from tracks with no later
same-ID observation. I then refit the same model on the more densely
labeled OpenDD cohort. It beat OLS4 there but lost to CV, which is the
stronger reference on that cohort.
[Raw-label audit](outputs/publication_readiness_2026_09/moving_future_support_v1/README.md) |
[OpenDD refit](outputs/publication_readiness_2026_09/opendd_weighted_velocity_v1/README.md).

I then trained a small forecast edit against the direction of error
change at CV. At the same recipients and displacement radii it beat
the earlier learned shape, but fewer recordings passed the risk
check. Keeping all edits while reallocating a smaller budget added
little utility and did not recover the missing easy references.
[CV-tangent experiment](outputs/publication_readiness_2026_09/opendd_cv_tangent_v1/README.md) |
[Updated paper](outputs/publication_readiness_2026_09/evidence_manuscript_v50/README.md).

I next trained the model to limit the easy-case error added by the
actual finite edit. It recovered 21 recording-seed passes without
losing any the tangent model passed, but gave up forecast accuracy.
Even picking the best of the new paths with future labels could not
close the gap at the same recipients and radii.
[Finite-harm continuation](outputs/publication_readiness_2026_09/opendd_finite_guard_v1/README.md).

I checked four public SinD city samples and the Foggy Bottom pedestrian
release for the same 8-step observation and 12-step prediction setup.
On the roughly 0.4-second grid, 88.6% of the selected SinD targets have
all 12 labels, versus 36.0% in Foggy Bottom. I am checking how the
published positions were produced before using either release to test
causal forecasts. [Source audit](outputs/publication_readiness_2026_09/independent_source_qualification_v1/README.md).

I have since checked ACME's public release. Seven site samples give
1,202 nonoverlapping 8-to-12 windows with consistent frame labels.
I also reproduced a camera-only detector and tracker on one Michigan
clip. Its published annotation points do not align reliably with the
detector boxes under a direct frame-and-pixel match, so I have not scored
those forecasts. [ACME audit](outputs/publication_readiness_2026_09/acme_prefix_intake_v1/README.md).

On the Michigan clip, the frozen detector tracked a wall portrait and
missed people farther down the blurred corridor. Filtering to the floor
removed those persistent detections. Higher-resolution crops and a
past-only foreground method still recovered no verified 8-step pedestrian
history across the two checked segments, so I did not use this camera to
validate forecasts. [Camera repair experiment](outputs/publication_readiness_2026_09/acme_camera_repair_v1/README.md).

At George Mason, a separate ACME camera yielded more usable histories.
I kept the detections fixed and changed how tracks are linked using
past motion; that recovered an eighth matched history. Five had all
twelve future labels. OLS4 reduced average path error relative to CV
on those five, but endpoint error hardly changed. This is an early
camera diagnostic, not a cross-site result.
[GMU readout](outputs/publication_readiness_2026_09/acme_gmu_camera_v1/README.md).

I then ran the same detector and association code on two unused ACME
cameras. The association change did not recover more stable histories
at CMU or Miraikan. CMU had seven complete targets and OLS4 again beat
CV on average path error; Miraikan did not have enough matched histories
to score. [Frozen camera comparison](outputs/publication_readiness_2026_09/acme_held_camera_probe_v1/README.md).

I kept those windows fixed and replaced the detector with a stronger
pretrained model. Stable histories increased from 8 to 10 at CMU and
from 1 to 9 at Miraikan, so both cameras now support a small baseline
comparison. I have only 17 complete targets across the two cameras;
I treat this as an input repair, not independent evidence for the
forecasting policy. [Detector comparison](outputs/publication_readiness_2026_09/acme_detector_hybrid_v1/README.md).

I took the frozen OpenDD forecaster to TUMDOT-ING without fitting on the
new recordings. Its 5% blend failed to improve on CV or on a causal
motion control at the same recipients and coefficient. It met the
original recording risk rule in only 19 of 27 recording-seed views.
[Transfer test and failure analysis](outputs/publication_readiness_2026_09/tumdot_ing_frozen_transfer_v1/README.md).

I revisited those same recordings with all methods editing the same
trajectories by the same maximum displacement. The new neural direction
slightly improved on my earlier model, but still lost to CV and to a
past-motion backtest. Smaller edits passed all 27 recording-seed safety
checks without improving on CV. Looking at the future showed that the
available paths could have improved forecasts safely; the frozen model
could not tell when to use them from past motion alone.
[Matched transfer experiment](outputs/publication_readiness_2026_09/tumdot_matched_finite_transfer_v1/README.md).

I trained a small model on OpenDD to choose among those six paths,
then took it to the same TUMDOT recordings without fitting there. The
model with explicit history did worse than its path-only version, and
neither beat the simple motion baselines. At the larger matched radius,
six checks failed on easy cases at one intersection. I kept the existing
policy and left the independent DroneCrowd groups sealed.
[Learned path-choice experiment](outputs/publication_readiness_2026_09/matched_path_choice_v1/README.md).

I also matched the training candidates to the averaging used at
inference. Averaging changed which paths the model selected, but the
history-based version still passed only 21 of 27 risk checks and lost
to the motion baselines. I did not replace the current policy.
[Matched averaging experiment](outputs/publication_readiness_2026_09/matched_path_ensemble_v1/README.md).

I then asked whether shorter parts of the observed tracks could tell me
which future direction to use. On OpenDD this beat the simple historical
control, but assigning the same number of each path at random did better
than the linear replay rule. The result did not carry over to TUMDOT.
[History replay and matched-count comparison](outputs/publication_readiness_2026_09/prefix_path_replay_v1/README.md).

I also tested whether the published OpenDD road maps help choose a
forecast. Continuing a pedestrian's recent motion relative to the road
boundary gave a small gain over a shuffle within similar map conditions,
but the full policy still lost to the fixed tangent forecast and failed
the original risk rule. I kept the deployment unchanged.
[Road-map comparison](outputs/publication_readiness_2026_09/opendd_map_direction_v1/README.md).

I then trained the same path chooser with no map, the correct map,
or map features exchanged between agents observed at the same time.
The correct map changed decisions, but it did not clearly improve on
either control or on the fixed tangent forecast. I did not replace it.
[Learned map comparison](outputs/publication_readiness_2026_09/opendd_map_path_value_v1/README.md).

I tested the frozen OpenDD models on Hannover shared-space tracks for
the first time. The small neural changes improved on constant velocity,
but OLS4 was stronger on average. Even an optimistic calculation for
the fixed CV-centered budget could not reach OLS4's aggregate lower
bound. Missing-label charges alone prevented the original risk rule from
passing. I did not use Hannover to tune a replacement policy.
[Hannover transfer and capacity analysis](outputs/publication_readiness_2026_09/hannover_frozen_transfer_v1/README.md).

I then let the audit skip individual edits and choose blends toward
OLS4. With access to the future, the expanded bank could improve on
both motion references under the original risk rule in more OpenDD
views and all nine Hannover views. Clipping the new directions to the
old radius unlocked none. This is headroom for a different policy,
not a deployed model result.
[Optional bridge and matched-radius control](outputs/publication_readiness_2026_09/optional_reference_bridge_v1/README.md).

I trained a past-only model to use the expanded choices. It nearly
always fell back to constant velocity. Holding the edit budget fixed,
replacing the predicted risk with its hindsight value recovered many
more useful edits; replacing predicted gain did not. I did not treat
fallback as a safety pass or deploy the new policy.
[Learned bridge and risk diagnosis](outputs/publication_readiness_2026_09/reference_bridge_policy_v1/README.md).

I then trained the OLS4 bridge risk head to predict its margin to the
existing risk boundary directly. It intervened more often, but the
increase included too many incorrect approvals. It still lost to fixed
tangent and passed fewer checks than the independent rule at the same
switch count.
[Direct risk and admission audit](outputs/publication_readiness_2026_09/signed_bridge_risk_v1/README.md).

I next calibrated the actions that the policy actually chose, using
separate development locations. Adding an easy-support score did not
improve the result: both the joint policy and the independent rule at
the same switch counts cleared 54 of 606 recording-seed evaluations.
With the same candidates and edit cap, a hindsight search found useful
and risk-feasible changes in 170 of 202 distinct recordings. The bank
has room to improve; the learned ranking did not find it reliably.
[Calibration and capacity study](outputs/publication_readiness_2026_09/selected_bridge_hurdle_v1/README.md).

I then kept the number of changes per query fixed and trained the model
to compare alternative recipients and amplitudes directly. Its average
gain over independent choice was uncertain, and a simpler additive
model cleared more recording checks. At those exact counts the
hindsight ceiling fell to 276 of 606 recording-seed evaluations, so
both where to intervene and what to change still matter.
[Same-count swap study](outputs/publication_readiness_2026_09/matched_bridge_swaps_v1/README.md).

I next trained a separate model to decide how many trajectories to edit.
The recording-level teacher led to fewer edits under the same online
budget, but it did not improve forecast gain. When I matched the exact
number of edits per recording in a retrospective check, the simpler
query-local model gave better gain. I kept the existing policy.
[Recording-aware count study](outputs/publication_readiness_2026_09/recording_count_v2/README.md).

I separated the limits of the nominated trajectories from the limits of
when a policy can spend its edit budget. The strictest hindsight search
still found useful, risk-feasible edits in 330 of 606 recording-seed
views. Retraining on plans that obeyed the exact online budget did not
improve held-location results over the projected-plan control. I kept
the existing policy.
[Capacity audit](outputs/publication_readiness_2026_09/streaming_menu_capacity_v1/README.md) |
[Teacher comparison](outputs/publication_readiness_2026_09/token_count_teacher_v1/README.md).

I then learned the gain and harm of each candidate edit directly.
Adding two nearby agents' histories did not improve the comparison
at equal edit counts: the extra context gave fewer joint passes, and
the forecast gain interval crossed zero. The main draft now shows both
the feasible action capacity and the limits of learning to use it.
[Direct value study](outputs/publication_readiness_2026_09/direct_count_value_v1/README.md) |
[Working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v66/README.md).

I then separated gain, harm and easy-reference support at the same
recording edit counts. True harm improved more pass counts than true
support, while true gain gave a clearer accuracy benefit. Estimating
harm and reference mass separately improved the matched pass count
from 60 to 73 of 606 views, but did not improve forecast utility and
overcalled positive easy risk. I did not change deployment.
[Attribution](outputs/publication_readiness_2026_09/support_value_attribution_v1/README.md) |
[Factorized risk](outputs/publication_readiness_2026_09/factorized_count_risk_v1/README.md).

I then calibrated risk using predictions from locations excluded from
both model fitting and candidate generation. Fewer false alarms at a
fixed probability threshold came with almost no true easy-risk detections.
The soft score cleared two more matched views but fewer online, and
lower gain prediction error did not yield better decisions.
[Donor-only calibration study](outputs/publication_readiness_2026_09/donor_risk_readout_v1/README.md).

I also trained the model to compare edits that compete for the same
recording budget. The swap loss cleared more joint checks than point
regression in the retrospective match, but ranked fewer certain swaps
correctly and lost forecast gain. It also cleared fewer checks online.
[Cross-query swap study](outputs/publication_readiness_2026_09/recording_swap_rank_v1/README.md).

I then changed the candidate trajectories while keeping every selected
agent, prediction time and edit radius fixed. The wider motion menu
increased hindsight capacity, and a learned trajectory proxy improved
on the straight edit and a previous neural shape at the same radii.
It did not establish an advantage over a simple prefix backtest and
could not fix missing easy support.
[Candidate geometry](outputs/publication_readiness_2026_09/radius_matched_turn_menu_v1/README.md) |
[Learned shape choice](outputs/publication_readiness_2026_09/shape_menu_value_v1/README.md) |
[Working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v70/README.md).

I then kept the edit times, counts and radii fixed and changed which agents
received them. With the outcomes known, more recordings could meet both the
risk and accuracy checks. The causal policy did not recover that advantage,
and over 250 recording-seed views had no eligible way to obtain an easy
reference at those edit times. I kept the earlier policy unchanged.
[Matched recipient experiment](outputs/publication_readiness_2026_09/recipient_slot_v1/README.md).

I next kept each recording's edit count and displacement radii fixed but
allowed edits to move to later queries. In hindsight, a two-query delay
raised the joint witness count from 206 to 298. My real-time controller did
not recover that gain, even when trained to compare waiting directly
with the original edit. Some recordings could never earn enough credit
for one edit under the fixed budget; I report those as undefined, not safe.
[Timing experiment](outputs/publication_readiness_2026_09/slot_timing_v1/README.md) |
[Learned wait decision](outputs/publication_readiness_2026_09/release_wait_v1/README.md) |
[Updated paper](outputs/publication_readiness_2026_09/evidence_manuscript_v72/README.md).

I also trained the wait decision on the actual recording schedule rather
than treating each edit in isolation. This changed some training labels,
but did not produce a reliable gain. With the future known, the same
decoder could pass 138 checks instead of 77 by changing at most one
decision per recording without dropping edits. The remaining problem
is finding those changes from observed motion, not merely simulating
the budget more faithfully.
[Whole-recording experiment](outputs/publication_readiness_2026_09/ledger_wait_v2/README.md) |
[Updated paper](outputs/publication_readiness_2026_09/evidence_manuscript_v73/README.md).

I then compared observed peer histories against self-only and
wrong-focal histories from the same query. The correctly aligned peers
did not reliably improve the decision. Replaying the same emitted
edits showed that the full policy's loss came almost entirely from
giving up an immediate edit and failing to find a replacement in time.
[Peer ablation and expiry replay](outputs/publication_readiness_2026_09/wait_peer_value_v1/README.md) |
[Supplement](outputs/publication_readiness_2026_09/evidence_manuscript_v74/README.md).

I separated the value of a successful wait from the cost of losing the
original edit. The asymmetric cost reduced expiry loss substantially,
but also made fewer useful moves and did not establish better overall
performance. I kept the original policy. The accompanying offline
replay recomputes the paired intervals without releasing trajectory data
or checkpoints.
[Expiry pricing](outputs/publication_readiness_2026_09/expiry_pricing_v1/README.md) |
[Numerical replay](outputs/publication_readiness_2026_09/deferral_review_replay_v1/README.md).

I then let the controller advance its first edit and repay it from
later credit. It passed more development checks, but did not reliably
reduce forecast error. In the 105 views without enough credit,
only 30 could beat both baselines safely even with hindsight and a
free choice of one candidate. The causal policy recovered three.
[First-credit experiment](outputs/publication_readiness_2026_09/first_credit_v2/README.md) |
[Working supplement](outputs/publication_readiness_2026_09/evidence_manuscript_v76/README.md).

I also tested whether holding a turn longer improves those
candidates. It worsened forecasts even at the same edit radii.
An exact hindsight calculation shows that even unconstrained
one-edit paths can pass in only 60 of the 105 credit-starved views.
[Turn and one-edit capacity experiment](outputs/publication_readiness_2026_09/sustained_turn_v2/README.md) |
[Working supplement](outputs/publication_readiness_2026_09/evidence_manuscript_v77/README.md).

I then held the intervention count fixed at two and compared
independent and joint choices from the same value model. The joint
choice did not reliably reduce error and passed fewer of the
recording checks. A second edit adds only two distinct feasible
recordings to the short-roster hindsight comparison.
[Matched two-edit experiment](outputs/publication_readiness_2026_09/two_edit_v1/README.md) |
[Working supplement](outputs/publication_readiness_2026_09/evidence_manuscript_v78/README.md).

I also held the first edit fixed and moved the second into a later
window. That worsened forecasts in the recordings with two edits in
both schedules. Using the first score to inform the second choice,
with both windows fixed, did not reliably improve accuracy.
[Cross-window comparison](outputs/publication_readiness_2026_09/cross_query_edit_v2/README.md) |
[Working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v79/README.md).

I trained a head to predict whether the first edit would pass the
whole-recording check. It recovered passes relative to its own
moment readout, but did not beat the frozen control. The same
one-edit bank has much more hindsight capacity when the controller
is not forced to act in the first eligible window.
[First-edit supervision](outputs/publication_readiness_2026_09/first_edit_target_v1/README.md) |
[Working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v80/README.md).

I then tested whether an online rule could wait for a better first
edit. It passed more recording checks but increased conservative
error even when both policies made one edit. Requiring both donor
criteria mostly sent the rule back to immediate editing.
[One-edit stopping comparison](outputs/publication_readiness_2026_09/one_edit_stopping_v1/README.md) |
[Working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v81/README.md).

I also checked four historical CREATE jobs against the scheduler and
matched the remote freeze manifest to my local copy. That confirms the
archived run lineage; the earlier development result remains negative.
[CREATE verification](outputs/publication_readiness_2026_09/create_accounting_recovery_20261108/README.md).

On OpenDD I also retrained the forecast around the motion blend that the
policy actually emits. It improved on hindcast, but failed the recording
risk test. At equal realized displacement it still beat hindcast slightly
and lost to my previous neural forecast, so I did not replace that model.
[Training and matched-displacement comparison](outputs/publication_readiness_2026_09/opendd_hindcast_residual_v1/README.md).

I then fixed the actual displacement budget inside training. The
retrained forecast was slightly better than the previous neural model
at every held OpenDD location, but it hurt enough easy trajectories
to pass fewer recordings. Adding neighboring pedestrians' past
tracks did not give a reliable gain.
[Fixed-budget training and risk analysis](outputs/publication_readiness_2026_09/opendd_radius_shape_v1/README.md).

I trained directly against the recording risk next. It passed more
recordings but lost forecast accuracy against the previous model.
Even a hindsight calculation could not pass every recording at the
fixed intervention budget, which is why I am also testing where and
how far to intervene.
[Recording-loss experiment and bound](outputs/publication_readiness_2026_09/opendd_flight_shape_risk_v1/README.md).

I compared online allocation with spreading the same movement over
more pedestrians. Spreading recovered many risk passes without
increasing total movement, and a fixed half-amplitude rule was a
strong control. The joint rule did not improve on simpler selection
at equal realized budget. None of these passed every recording.
[Matched movement comparison](outputs/publication_readiness_2026_09/opendd_causal_amplitude_v1/README.md).

I also tried moving the fixed-half budget between pedestrians without
dropping any target. Both learned guards kept its recording passes,
but neither reliably improved prediction. Knowing the future shows
there are useful exchanges; the models did not reliably identify them
from the past.
[Marginal and direct exchange results](outputs/publication_readiness_2026_09/opendd_direct_exchange_v1/README.md).

I then tested time-aligned traffic history against the same summaries
reassigned within each query. Both gave tiny gains over fixed half, but
aligned traffic did not beat the pedestrian-only or shuffled controls.
I report this as a negative context ablation in the revised main draft.
[Context experiment](outputs/publication_readiness_2026_09/opendd_context_exchange_v1/README.md) |
[Updated manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v35/README.md).

I also checked two public aerial datasets for a separate trajectory study.
AerialMPT has dense annotations but only eight non-overlapping clips or
segments with complete multi-agent 8-to-12 support. OpenDD has much more
complete support, but two roundabouts account for most of it.
[Source audit](outputs/publication_readiness_2026_09/topdown_source_intake_v1/README.md).

On OpenDD I trained a gain-and-harm selector with one roundabout held out
at a time. It beat a matched history-based heuristic, but last-displacement
CV was stronger than either learned selector. Adding nearest-neighbor
history made the selection worse, and neither model protected easy cases.
[OpenDD experiment](outputs/publication_readiness_2026_09/opendd_location_diagnostic_v1/README.md).

I then made CV the default on OpenDD and let the model abstain. A second
reference and location-held calibration kept easy errors almost unchanged,
but the model intervened too rarely to establish a gain over CV. A hindsight
comparison showed that the same motion menu still had room for improvement.
[CV-anchored follow-up](outputs/publication_readiness_2026_09/opendd_cv_abstention_v1/README.md).

Next I held the motion forecasts and number of edits fixed and trained
the models to rank edits within each query. The new loss did not beat
pointwise gain regression. Holding either the person or the candidate
fixed exposed room for improvement in hindsight, but not a reliable
learned gain. [Matched ranking and attribution](outputs/publication_readiness_2026_09/opendd_query_rank_v1/README.md).

I then changed the forecast itself. A fixed blend of the self-history
velocity model and CV improved forecast error at all seven OpenDD
locations, but increased easy-case error too much at one site. Adding
past traffic context did not help in this comparison.
[Direct forecast experiment](outputs/publication_readiness_2026_09/opendd_velocity_candidate_v1/README.md).

I reduced the blend using only other locations. The resulting forecast
still improved on CV and on two older forecasts at the same intervention
counts, but failed the recording-level risk checks. For some recordings
the easy reference was absent; for others the selected changes were too
costly. [Nested calibration](outputs/publication_readiness_2026_09/opendd_velocity_calibration_v1/README.md).

I then let the selector reserve predicted risk as a recording unfolds.
It reduced easy-case harm in aggregate, but fewer recordings passed the
original checks. At equal edit counts, joint geometry reduced a distance
proxy relative to independent selection and lost trajectory accuracy.
[Streaming allocation](outputs/publication_readiness_2026_09/opendd_prefix_risk_v1/README.md).

I kept the switch counts fixed and tested whether the policy could retain
easy references needed for the risk calculation. A simpler estimated
reference-mass ratio worked better than my new classifier on recording
passes. Neither satisfied the full recording requirement.
[Matched easy-support comparison](outputs/publication_readiness_2026_09/opendd_easy_support_v1/README.md).

I also looked beyond OpenDD for better-labeled tracks. Three initial
TUMDOT-ING recordings retain 86% complete future paths on the audited
sampling grid. The railway archive is much larger, but its sampled
tracks retain only about 3%. That intake did not score either source.
[New source audit](outputs/publication_readiness_2026_09/station_sharedspace_intake_v1/README.md).

I followed up by learning when to replace OLS4 from short backtests within
the observed history. The method helped on available labels but did not
beat OLS4 conservatively. Shrinking all frozen replacements together did
not fix that. [OLS4 replacement study](outputs/publication_readiness_2026_09/native_ols_replacement_v1/README.md).
Estimating which future labels would be available improved that prediction
task, but did not improve replacement decisions at equal counts.
[Support-aware selection](outputs/publication_readiness_2026_09/native_ols_support_selection_v1/README.md).
I checked the activation scores on held-out fitting sources. None of the
predeclared thresholds showed sufficiently consistent gain to carry forward.
[Source-held calibration](outputs/publication_readiness_2026_09/native_ols_crossfit_calibration_v1/README.md).
Bounding the predicted gain by trajectory disagreement favored larger
replacements. Matching their amplitudes reduced harm but still did not
establish an advantage over direct prediction.
[Bounded gain and amplitude controls](outputs/publication_readiness_2026_09/native_ols_bounded_gain_v1/README.md).
I also tested the forecast itself, separating its starting position from
its velocity estimate. Fixed position smoothing was worse than OLS4, and
the learned variant nearly reverted to it.
[Forecast-anchor comparison](outputs/publication_readiness_2026_09/native_ols_anchor_denoising_v1/README.md).
I then held the correction size fixed and let each method pick one agent
per query. Short history backtests helped identify potential beneficiaries,
but the learned choices still lost to OLS4. Even hindsight selection gained
only 0.0504 points on the CV-normalized score at that budget.
[Matched anchor selection](outputs/publication_readiness_2026_09/native_ols_anchor_retrieval_v1/README.md).
I next changed the forecast itself, learning velocity adjustments from
past neighbor motion. The near-neighbor model did not beat matched
outer-neighbor or self-only controls on the full prediction grid. A
simple fixed rule helped on visible labels, but its advantage did not
survive accounting for every missing step.
[Local velocity experiment](outputs/publication_readiness_2026_09/native_local_velocity_v1/README.md).
I held the model fixed and separated two parts of its loss. Removing
the extra harm weight let the model move and helped on complete tracks,
but did not improve the conservative full-grid result. With one edit
per query, hindsight choice still found some useful forecasts; random
choice lost ground. [Matched loss experiment](outputs/publication_readiness_2026_09/native_velocity_objectives_v1/README.md).
I trained the selector again with each evaluation source excluded from
both training stages. Its learned choices did not improve on leaving
OLS4 alone. Keeping those same choices and falling back to CV did not
fix them either. [Nested selection experiment](outputs/publication_readiness_2026_09/native_nested_velocity_selection_v1/README.md).
I then trained the ranker against CV itself. A fixed quarter-amplitude
diagnostic gave a small gain over no edit, but did not establish an
advantage over the matched simple controls or protect easy cases.
[CV ranking experiment](outputs/publication_readiness_2026_09/native_cv_query_rank_v1/README.md).

I also trained a mixture of five motion forecasts, using short backtests
within each observed history to decide which forecasts to trust. It
reduced error on available labels, but the advantage over OLS4 did not
hold when every missing prediction step was counted. The comparison
with the model without backtests leaves their additional value unproven.
[Motion mixture and matched controls](outputs/publication_readiness_2026_09/native_replay_motion_v1/README.md).

I then kept that forecast menu fixed and learned which two agents to
change in each query. It did not outperform leaving OLS4 alone or
prove better than choosing the smallest changes. The joint term
reduced forecast-distance contraction but did not improve the paired
forecast score. Most of the gap to hindsight came from choosing
which agents to change, not which forecast to use for them.
[Expert allocation experiment](outputs/publication_readiness_2026_09/native_expert_risk_allocation_v1/README.md).

I isolated that recipient choice by giving every agent one fixed
candidate and training only the ranking. Optimizing the expected gain
of two choices helped on sources seen during training, but not when I
held a source out. I also separated the loss of available gain from
restricting the candidate menu and shrinking the forecasts.
[Fixed-candidate ranking](outputs/publication_readiness_2026_09/native_recipient_top2_v1/README.md).

I then tested what the past bounding boxes could tell me that their
centers could not. Box and boundary cues made future label availability
much easier to predict across held sources. That improvement did not
carry over to a reliable gain over OLS4 or the smallest-change control.
[Observation-quality experiment](outputs/publication_readiness_2026_09/native_observability_v1/README.md).

I froze those label-support estimates and trained only the gain head.
Comparing agents on the steps both have labeled did not beat ordinary
regression under matched support. I keep the support result separate
from the unresolved forecast decision.
[Conditional-gain experiment](outputs/publication_readiness_2026_09/native_conditional_gain_v1/README.md).

I tested whether the observed video could add what the trajectory
history was missing. I aligned the complete past-frame set, measured
both background and local pixel motion, and kept the forecasts and
support budgets fixed. Neither image cue improved the matched
intervention scores.
[Past-pixel motion experiment](outputs/publication_readiness_2026_09/native_pixel_gain_v1/README.md).

## Looking For Better Observed Scenes

I audited two Waterloo drone sites before using them for forecasting.
The trajectory exports provide substantial twelve-step support, but
the site with the most complete longer windows also has many stationary
histories. The released positions come from an offline export whose
use of later frames has not been established. I keep these data out of
the causal confirmation experiment.
[Waterloo source audit](outputs/publication_readiness_2026_09/waterloo_annotations_v1/README.md).
On a separate Waterloo roundabout recording, I ran a detector and tracker
directly from unannotated frames, and checked that stopping early gives
the same past detections, track assignments and forecasts.
[Raw-video prefix experiment](outputs/publication_readiness_2026_09/waterloo_raw_prefix_v1/README.md).
I then tried to align a matching video and annotation export with a
municipal orthophoto. Registering each frame through the first frame
worked over the short prefix, but not over most sampled keyframes.
Finer detector tiles found more person-class boxes without recovering
the labeled pedestrians. I am not using the recording for forecast scores.
[Map alignment and detector comparison](outputs/publication_readiness_2026_09/waterloo_ortho_registration_v1/README.md).

## Earlier Development Result

I tested whether adding past-image alignment to each agent's history helps
without replacing its native coordinates. The new model keeps both views;
the matched control adds the same capacity without the camera transform.
There is no forced future-camera restoration or trajectory residual decoder.

| Whole-12 Comparison | Known-Label Gain Difference | Completion Lower | Source-Bootstrap Interval for Lower |
|---|---:|---:|---:|
| Observed-view node vs same-capacity native node | +0.828 pp | -4.979 pp | [-7.692, -2.333] pp |
| Observed-view node vs original actor | -0.321 pp | -5.995 pp | [-8.461, -3.530] pp |

**Neither required comparison passes.** The same-capacity control itself loses
1.148 points against the original. The new model's gain against the original
is +1.948 points at one step, but it does not persist over all twelve steps.
Known-easy positive harm still exceeds 2% in 33 of 34 raw-actor recording
diagnostics. This is not a complete selected-policy certificate. I retain the
original model and evaluation rules, with no new controller or deployment.

All twelve treatment fits completed 4,000 updates. The full shared-core raw
control and both fixed-design refits match exactly. A disk guard saved a
coherent step 3,935 checkpoint; the resumed model performed exactly the
remaining 65 updates. All twelve new forecast banks and scalar results replay
exactly. Eighteen existing controls are verified cache, not fresh inference.

This is one forecaster seed on 125,540 previously exposed donor windows from
14 source groups. I observe 8 steps and predict 12 at native stride 1 in image
coordinates. The completion lower bounds wholly unknown rows with the observed
reference fixed; it is not full-grid ADE for partial tracks. This does not
establish independent confirmation, calibrated time/scale, true 3D, a
foundation world model, or submission readiness.

[Full experiment](outputs/publication_readiness_2026_09/moving_dual_view_actor_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_dual_view_actor_v1/failure_analysis.md),
[reproduction instructions](outputs/publication_readiness_2026_09/moving_dual_view_actor_v1/operation_zh.md).

### Which Forecasts To Change

I also trained a model to propose at most four same-query exchanges per
recording. After calibration, it did not add or lose any of the original
15 passing views. It recovered an easy reference in one view without
making that view pass. I kept the original roster.
[Learned sparse exchanges](outputs/publication_readiness_2026_09/moving_sparse_quarter_exchange_v1/readout.md).

I checked whether a smaller intervention could preserve the original
utility and meet the same recording risk rules. In a hindsight-only test,
one same-query exchange repaired 26 previously failing views without
lowering paired utility. This identifies room for a selective replacement
model; it is not a forecast result or a change to the deployed policy.
[Sparse exchange experiment](outputs/publication_readiness_2026_09/moving_quarter_swap_capacity_v1/readout.md).

I trained a model to predict the gain and harm of each candidate, then let
it change the selected agents while keeping every query's intervention
count and amplitude fixed. Adding the harm term reduced total-risk
violations from 64 to 56 out of 102 views compared with gain-only ranking.
Both learned rankings passed just 7 views, versus 15 with the original
agent list at quarter amplitude. The gains on labeled cases did not
survive the missing-label bounds. I kept the existing policy.
[Matched-count allocation experiment](outputs/publication_readiness_2026_09/moving_quarter_allocation_v1/readout.md).

### How Much To Change A Forecast

I kept all 6,952 neural interventions fixed and changed only how far each
forecast moved away from constant velocity. Using one quarter of the
original adjustment raised recording passes from 2 to 15 out of 102.
A learned adjustment that shared the same budget within each query
raised that count to 19, but did not show a reliable gain over the fixed
quarter. The donor-calibrated model almost always chose the fixed quarter.
The original 2% rule still applies, 35 views lack a required reference,
and none of these variants is deployed.
[Amplitude study and exact replay](outputs/publication_readiness_2026_09/moving_calibrated_amplitude_v1/readout.md).

### Fixed-Forecaster Decision Study

I also scored the frozen policies on the relative motion of neighboring
agents. Joint selection beats independent selection on labeled pairs, but
does not establish a gain over the matched additive control once missing
pair steps are accounted for. The simple OLS4 forecast has lower observed
pair error even though its scene ADE is higher. I report both metrics.
[Joint path audit](outputs/publication_readiness_2026_09/moving_joint_path_audit_v1/readout.md).

I tested that difference by holding OLS4 relative paths fixed and changing
only their shared motion. A fixed uniform mixture improves the lower
utility bound against OLS4 by 8.503 percentage points with the CV reference
denominator, while preserving every pair vector. The learned mixture does
not reliably beat the fixed controls or CV; no full-scene variant passes
the recording risk rule.
[Common-motion ablation](outputs/publication_readiness_2026_09/moving_common_motion_v1/readout.md).

I compared additive selection with an exact pair interaction term, holding
the forecast candidates and each query's intervention count fixed. Across
14 excluded donor sources the added term did not improve relative-position
error. In a label-aware diagnostic, adding it to already known marginal
costs improves squared pair error by only 0.30%; the much larger gap is
between learned choices and the label-aware optimum. This motivates better
marginal cost estimation before a larger pair solver, not a deployment change.
[Decision study and reproduction](outputs/publication_readiness_2026_09/moving_pair_coupling_v1/README.md).

### Independent Evaluation Preparation

The [DroneCrowd calibration readout](outputs/publication_readiness_2026_09/dronecrowd_calibration_readout_v1/README.md)
keeps utility, the original recording risk checks, and confirmation
eligibility separate. It can be generated only after the full score
and replay.

I measured how much new labeling would matter for the fixed pair
comparison. Only 3,492 missing positions can change it. A targeted
order needs 1,691 valid positions to remove 95% of the original
disagreement debit, compared with 3,454 in chronological order.
[Conditional labeling budget](outputs/publication_readiness_2026_09/native_pair_annotation_budget_v1/README.md).

I constructed two bounded completions that preserve every released
coordinate. They rank joint and additive selection in opposite orders
on the same pair metric. This says what the current labels cannot
determine, not what the missing trajectories were.
[Constructive check](outputs/publication_readiness_2026_09/native_pair_completion_witness_v1/README.md).

I checked whether using one target for overlapping predictions of the
same agent pair changes the conservative geometry result. The bound
tightens, but the comparison with additive selection still does not
establish an advantage. [Shared-target analysis](outputs/publication_readiness_2026_09/native_shared_pair_completion_v1/README.md).
Grouping different edges that share a single unobserved agent
narrows it further without changing that conclusion.
[Observed-neighbor analysis](outputs/publication_readiness_2026_09/native_anchored_pair_completion_v1/README.md).

I measured the three-seed inference route on existing development
inputs. Nearly all compute is in the EqMotion forecast, not in
joint selection. The running external calibration keeps its original
settings. [Local timing and exactness check](outputs/publication_readiness_2026_09/frozen_inference_profile_v1/README.md).

I tried choosing velocity windows by how well they predict the
last few already observed positions. That helped slightly where
future labels are present, but it did not improve the conservative
12-step comparison or the recording risk pass count.
[Prefix forecast experiment](outputs/publication_readiness_2026_09/native_prefix_velocity_v1/README.md).

I also checked whether any reweighting of the same two forecasts
could satisfy the original recording rule. Of 102 recording-seed
views, 23 have no selected easy reference mass, 74 still exceed
the budget at the lower bound, and five admit a verified witness.
That points to changing who receives a new forecast, or which
forecast they receive, rather than fitting another mixture of
these same paths. [Recording feasibility](outputs/publication_readiness_2026_09/native_recording_blend_feasibility_v1/README.md).

I used a convex directional test to check the full range of
pair blend weights at fixed recipients. Separate weights make
more individual queries improvable than a shared weight, but
the assembled witnesses still fail the original recording
requirement. [Directional audit](outputs/publication_readiness_2026_09/native_pair_blend_direction_v1/README.md).

I trained a blend that charges disagreement on missing future
labels. It mostly falls back toward the individual forecast:
the typical pair-correction weight drops from 0.412 to 0.058.
It does not improve the original recording pass count.
[Matched blend experiment and figure](outputs/publication_readiness_2026_09/native_completion_motion_blend_v1/README.md).

I then trained the same neighbor model with and without a
relative-path loss. It reduced pair error on observed steps,
but the full-grid lower comparison was negative and the
recording pass count did not change.
[Matched training experiment](outputs/publication_readiness_2026_09/native_relative_path_training_v1/README.md).

I compared ego motion, pooled neighbor motion and local relative
features using the same switched people. Adding the local features
did not help either conservative utility or relative-path accuracy
under the original comparison.
[Context ablation](outputs/publication_readiness_2026_09/native_neighbor_motion_adapter_v1/README.md).

I trained a matched pair of OLS4-relative motion heads across
three seeds on disjoint training sources. The new forecasts helped
the all-row comparison with OLS4 on available labels but lost at
the original intervention recipients. Neither expanded the number
of recording views that could beat both motion references at the
same budget. [Training, recovery and comparison](outputs/publication_readiness_2026_09/native_gated_motion_residual_v1/README.md) |
[Optional-menu audit](outputs/publication_readiness_2026_09/native_gated_motion_menu_v1/README.md).

I tested whether extrapolating acceleration and turns could add
useful paths without changing who gets switched. Neither bank
extension added a positive view, and the matched speed and heading
controls made the actual predictions worse. [Motion-shape test](outputs/publication_readiness_2026_09/native_damped_motion_v1/README.md).

I separated the missing-label charge in the risk check from the
charge in the gain comparison. In the most permissive diagnostic,
15 additional views can pass, but 36 still fail. I keep the original
rule for all reported deployment evidence. [Debit ablation and its limits](outputs/publication_readiness_2026_09/native_missing_debit_capacity_v1/README.md).

I tried expanding the forecast bank with small OLS4-centered blends.
Neither the causal blends nor the learned blends added a single
recording view that could beat both motion references and meet the
original risk rule. [Matched-count blend test](outputs/publication_readiness_2026_09/native_ols_blend_capacity_v1/README.md).

I also asked whether a better assignment could beat both CV and
OLS4 at exactly the same switch counts. The full bank meets the
risk rule in 96 of 102 development views, but only 51 can also
beat both references under the conservative utility comparison.
Changing the hindsight optimizer's objective did not add a single
view. [Exact dual-reference result](outputs/publication_readiness_2026_09/native_dual_reference_capacity_v1/README.md).

I checked whether missing future labels explain why the imitated
policy failed. In 98 of 102 views, an original 2% check already
failed on observed labels. The missing-label bound makes the utility
comparison uncertain, but it is not the main reason the recording
checks fail. [Risk decomposition and supplement](outputs/publication_readiness_2026_09/evidence_manuscript_v14/supplement_readout.md).

I used feasible recording-level decisions to teach a past-only
ranker. It more often chose the same person as the teacher, but
that did not translate into safer decisions. At the same 9,675
switches, the full-action ranker lost 16.61 percentage points in
conservative utility relative to gain ranking. The original risk
screen kept no switches. I kept this negative result in the
[working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v14/README.md)
and left the independent evaluation protocol unchanged.

I tested a new way to choose recipients without changing their
nominated paths or the intervention budget. It does not beat the
gain control. Yet at exactly the same per-query counts, the
label-aware solver finds feasible choices in 93 of 102 views using
those same nominations. The gap is in finding those assignments
from past observations, not just in having more forecast paths.
[Past-only comparison](outputs/publication_readiness_2026_09/native_recipient_shortfall_v1/README.md) |
[Exact capacity and figure](outputs/publication_readiness_2026_09/native_exact_count_joint_capacity_v1/README.md).

I matched the number of neural replacements before testing a model
for missing future labels. It did not improve conservative utility.
I then let a label-aware audit pick the best option for each risk
component. Keeping the same recipients rules out 99 of 102 views
at the matched count, and 93 even when the neural count is free.
[Matched test](outputs/publication_readiness_2026_09/native_visibility_increment_v1/README.md) |
[Why reranking is insufficient](outputs/publication_readiness_2026_09/native_fixed_recipient_risk_bound_v1/README.md).

I trained a new score for replacing an already selected motion
forecast. At the same recipients it worsens the conservative
comparison by 3.303 percentage points against the previous score.
Most of that difference comes from greater disagreement on windows
with no future labels. I did not replace the existing policy.
[Increment test](outputs/publication_readiness_2026_09/native_direct_increment_v1/README.md) |
[Updated working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v13/README.md).

I then ran the same past-only selector with and without learned
forecasts, at exactly the same switch count. The mixed menu helps on
observed labels but does not establish conservative improvement.
On the queries both screens accept, the 56 switches are identical
and all use motion baselines.
[Matched causal-menu control](outputs/publication_readiness_2026_09/native_causal_menu_control_v1/README.md).

I separated the learned forecasts from the motion alternatives in the
same label-aware capacity test. The causal menu already accounts for
48 of the 51 positive recording-seed views. The learned forecasts
add three views from one recording, so I cannot credit the whole
gain to the neural models or treat it as a deployment result.
[Matched forecast-bank ablation](outputs/publication_readiness_2026_09/native_neural_bank_ablation_v1/README.md).

I pulled the decision study into a four-page working paper. The main
comparison keeps the prediction bank and intervention budget fixed. I
report where joint selection falls short, alongside the image ablation,
the recording-level risk refits, and a large easy-case failure.
[Working paper and reproduction](outputs/publication_readiness_2026_09/evidence_manuscript_v12/README.md).

I checked whether any one source drives the relative-motion result.
Joint selection improves observed pair error over independent selection
in all 14 source averages, but the added pair term does not have the
same support against the matched additive control. Its full-grid lower
estimate stays negative whichever source I omit.
[Risk definitions and source checks](outputs/publication_readiness_2026_09/evidence_manuscript_v12/supplement_readout.md).

I have fixed the DroneCrowd study to 15 calibration groups and 30
separate confirmation groups. The recordings stay with their entire
exclusion group. This assigns roles; it does not report external
predictions or open the confirmation labels.
[Approved split and checks](outputs/publication_readiness_2026_09/dronecrowd_approved_roles_v2/README.md).

I have now frozen the calibration candidates and started past-only
inference on the 40 calibration recordings. The 72 confirmation recordings
are excluded from the runner. The launch record does not contain outcomes.
[Calibration protocol and local run](outputs/publication_readiness_2026_09/dronecrowd_calibration_readout_v1/README.md).

I have also fixed how I will evaluate confirmation if calibration
qualifies. The comparison keeps the same switch count for the joint
and simpler choices, and separates forecast error from the original
recording risk check. The confirmation data are still sealed.
[One-shot protocol and tests](outputs/publication_readiness_2026_09/dronecrowd_confirmation_readout_v1/README.md).

I tested process-isolated CPU inference on previously exposed development
recordings. Parallel execution was 1.92 times faster on that benchmark
and left all tested predictions and switches unchanged. I have prepared
the optional confirmation runner without starting confirmation or
interrupting calibration.
[Exact scheduling benchmark](outputs/publication_readiness_2026_09/native_cpu_shard_benchmark_v1/README.md).

I checked what the frozen forecast options could achieve even with
hindsight. On 102 development recording-seed views, the four-option
slate was infeasible under the original risk rule in 41 views; only
27 views admitted a positive conservative gain over OLS4. This is
a limit of the tested options, not a result for a learned selector.
[Capacity analysis](outputs/publication_readiness_2026_09/native_joint_slate_capacity_v1/README.md).

I then kept the intervention recipients fixed and offered OLS4 and
a shared-motion variant on those same agents. The label-aware capacity
rose from 61 to 71 feasible views and from 27 to 33 with positive
conservative gain over OLS4. This tells me the forecast choices matter,
even with the same agents, but it is not a deployment result.
[Matched-recipient experiment](outputs/publication_readiness_2026_09/native_slate_geometry_capacity_v1/README.md).

I also trained a small, bounded forecast around OLS4. Choosing the
adjustment size on separate development sources helped against OLS4,
but adding nearest-neighbor information did not help the matched
comparison. The simpler forecast added no new passing recording views
when offered alongside the existing options, so I kept the policy unchanged.
[Local forecast and matched controls](outputs/publication_readiness_2026_09/native_local_ols_proposal_v1/README.md).

I then kept the query budget fixed and let the same forecast bank reach
more agents. In hindsight, this raised feasible views from 77 to 99
and views with positive conservative gain over OLS4 from 39 to 51.
It gives me a more specific target for retrieval, not a new trained
policy. [Fixed-budget recipient comparison](outputs/publication_readiness_2026_09/native_recipient_budget_capacity_v1/README.md).

I trained a past-only ranker to recover that opportunity. With eight
candidate recipients per query, it found three additional feasible
views but no new view with positive conservative gain over OLS4.
The original controller still passed more views, so I kept it unchanged.
[Learned retrieval comparison](outputs/publication_readiness_2026_09/native_full_query_retrieval_v1/README.md).

I also trained retrieval from whole-recording solutions. It found
three additional views with positive hindsight OLS4 utility, but
selecting one assignment per query and then applying the fixed gate
reduced that capacity from 48 to 26 to 3. The actual policy still did
not beat OLS4. [Where the choices were lost](outputs/publication_readiness_2026_09/native_recording_witness_retrieval_v1/README.md).

I compared all legal two-person path combinations on the same shortlist.
At 72 matched switches, the larger search changed one query and
did not improve the result. Screening all options first increased
switches but also increased recording risk violations.
[Exact joint search and additive control](outputs/publication_readiness_2026_09/native_joint_path_portfolio_v1/README.md).

I retrained risk prediction on the combinations the search actually
chose. With 3,280 switches each, the learned ranking and the matched
risk-margin control both passed one view. The new head predicted easy
reference better, but did not predict risk better.
[Source-separated post-search result](outputs/publication_readiness_2026_09/native_postsearch_risk_v1/README.md).

I compared query-level and whole-recording risk fits on the same
search outputs. At matched counts the whole-recording fit passed
one fewer view. I also traced the three hindsight-infeasible views
to one recording repeated across seeds: even its best easy replacement
adds 23.925% of the CV reference error.
[Risk fit](outputs/publication_readiness_2026_09/native_postsearch_mass_v1/README.md) |
[Failure case](outputs/publication_readiness_2026_09/native_easy_support_certificate_v1/README.md).

I used past motion to limit how far each forecast could move from CV.
With all 2,194 interventions held fixed, the tiny changes raise risk
passes from four to 40 but give up almost all of the gain over CV.
Requiring a material displacement changes none of the operating paths.
[Radius experiment](outputs/publication_readiness_2026_09/native_history_radius_v1/README.md) |
[Decision-focused working draft](outputs/publication_readiness_2026_09/evidence_manuscript_v11/README.md).

I compared predicted benefit and harm magnitudes with the previous
activation scores. At the same 430 switches, the risk-adjusted score
did not reliably improve utility. The current-query version passed
2 of 102 recording-seed views, so I have not promoted it.
[Continuous activation results](outputs/publication_readiness_2026_09/native_value_activation_v1/README.md).

I separated when to switch from which forecasts to use. At the same
430 switches, activating by predicted feasibility found more locally
feasible sets than activating by predicted gain, but reduced conservative
utility. The current-query version passed only 2 of 102 recording-seed
views. I have not promoted either rule.
[Matched activation and donor-only thresholds](outputs/publication_readiness_2026_09/native_activation_v1/README.md).

I trained a model to rank whole switch sets by their chance of being both
useful and within budget. At the same 222 switches, it did not improve
local feasibility and passed 2 rather than 5 recording-seed views.
The frozen switch schedule also left feasible alternatives unused. I am
keeping the old operating policy unchanged.
[Matched ranking experiment](outputs/publication_readiness_2026_09/native_feasible_slate_v1/README.md).

I tested whether changing the proposed switches could recover easy-case
coverage. At the same proposal count, the support-aware ranking raised the
number of views with a feasible hindsight choice from 55 to 67, but the
unchanged gate kept a feasible choice in only 18 views for either ranking.
It also worsened matched forecast accuracy. I have not promoted the policy.
[Matched proposal and gate audit](outputs/publication_readiness_2026_09/native_support_retrieval_v1/README.md).

I also trained a screen that judges a whole proposed switch set rather than
adding separate decisions. At nearly the same intervention count, it found
21 locally feasible sets instead of 10, but only 4 of 102 recording-seed
views passed the original rule. Every recording had easy-case reference
error somewhere; in 24 views, none of the proposed switches reached it.
I am keeping this as a development result, not a deployment claim.
[Switch-set experiment and capacity audit](outputs/publication_readiness_2026_09/native_query_risk_critic_v1/README.md).

I tested a new decision rule that only switches forecasts when the current
query predicts a gain, low risk and a positive easy-case reference. In development,
it reduced interventions from 9,675 to 90, but passed just 3 of 102
recording-seed views under the original risk rules. I have not promoted it
or opened the reserved DroneCrowd outcomes.
[Development rehearsal and matched controls](outputs/publication_readiness_2026_09/dronecrowd_transport_candidate_v1/README.md).

I checked the exact predictor bank against the existing DroneCrowd
confirmation reservation, and rebuilt the reference cuts from the
training annotations. The original nine neural checkpoints are covered.
Using the matching producer fit gives a candidate cut pair of 12.492
and 61.718 pixels, without reading DroneCrowd outcomes. This predictor
audit predates the approved split.
[Predictor and cut provenance](outputs/publication_readiness_2026_09/dronecrowd_exact_bank_v2/README.md).

I built a standalone review replay for the development tables. It
recomputes recording-risk counts and source-group intervals from
de-identified aggregate statistics without the training data,
weights, or this repository.
[Replay scope and instructions](outputs/publication_readiness_2026_09/review_replay_v1/README.md).

I now use the same past-only inference entry point for the causal baselines,
the kinematic model, and the learned comparators. It reproduces the
kinematic model's full development forecasts across three seeds and the
original probe batches for all 18 comparator checkpoints.
[Inference replay and remaining external decisions](outputs/publication_readiness_2026_09/dronecrowd_native_inference_rehearsal_v1/README.md).

I connected the cached DroneCrowd annotation prefixes to the existing 8/12
forecast input schema. All 112 recordings pass the three-prefix check, without
using future labels to select agents. The complete 5,376-pair screen against
the prior development image roster finds no match at the frozen thresholds.
The original release folders still have known internal overlap. I retain all
45 conservative exclusion groups in the approved calibration and confirmation
roles. Image matching is not proof of independent physical sites, and neither
check is an external prediction result.
[Source screen and proposed roles](outputs/publication_readiness_2026_09/dronecrowd_cross_archive_v1/README.md) |
[Input compatibility](outputs/publication_readiness_2026_09/dronecrowd_frozen_inputs_v1/README.md).

I also checked the existing three-seed forecast controls before scheduling new
training. All 18 trained checkpoints reproduce their saved predictions exactly
on 18,432 development probe rows. The prior image screen covers every contributing
training recording for 12 of those checkpoints. This prepares predictor controls;
it does not establish three-seed intervention performance or external accuracy.
[Three-seed control library](outputs/publication_readiness_2026_09/dronecrowd_three_seed_bank_v1/README.md).

I then held the trajectories and switch counts fixed and tested whether
three-seed disagreement improves the choice of when to switch. It reduced
held cost prediction error, but the conservative utility contrast was -3.53
percentage points (source interval -8.98 to -0.36). This change does not pass
and I do not use it for deployment.
[Matched disagreement experiment](outputs/publication_readiness_2026_09/moving_seed_dispersion_cost_v1/readout.md).

My follow-up charged the model for changing its decision when a future label
was unlikely to be available. It reduced the conservative loss to -0.22
percentage points against the same control, but the interval remained below
zero and easy harm was not reliably reduced. I keep this as a negative result.
[Support-aware decisions](outputs/publication_readiness_2026_09/moving_support_anchored_cost_v1/readout.md).
I also recovered the earlier full image screen for the ten additional
training recordings, so all 18 forecast controls now have source-screen
coverage. This is not external accuracy or verified geographic independence.
[Complete predictor coverage](outputs/publication_readiness_2026_09/dronecrowd_combined_forecaster_coverage_v1/README.md).

I replaced the whole-track read intermediate with an eight-frame input slice.
The full unassigned external input schedule now packs in a bounded stream:
31,472 queries and 4,520,912 past-supported target rows. The six matched Torch
predictors resume exactly on the older development probe. This verifies the
evaluation inputs and recovery path, not new accuracy or safety.
[Causal input stream](outputs/publication_readiness_2026_09/m3w_causal_forecast_stream_v1/README.md).

I then reran the original risk checks on each of the 34 development recordings
and three forecast seeds. EqMotion improves the known-label result over the
fitting-selected OLS4 baseline, but the conservative interval crosses zero.
No fixed-count switch variant passes the full recording checks. This separates
predictor accuracy from the case for deploying a switch policy.
[Recording-level comparison](outputs/publication_readiness_2026_09/moving_recording_forecast_replay_v1/readout.md).

I checked whether the fixed trajectories could satisfy the original budget
even with outcome-aware choices. With CV and EqMotion, 91/102 repeated views
are feasible at free count and 90/102 at the old fixed count. Adding the
other existing forecasts raises free-count feasibility to 101/102, still
short of the full roster. This is an offline limit, not selector performance.
[Candidate-bank feasibility](outputs/publication_readiness_2026_09/moving_recording_bank_ceiling_v1/readout.md).

I also trained a matched three-seed forecast ablation. An easy-case loss
reduced known easy harm by 6.98 percentage points, but the conservative
paired utility fell by 7.68 points and the remaining unsupported view did
not recover. The new forecasts are research candidates, not a deployment.
[Easy-aligned training experiment](outputs/publication_readiness_2026_09/moving_easy_aligned_forecast_v1/readout.md).

I then trained against disagreement on wholly unobserved fitting rows. The
new paths are closer to CV on the corresponding held slice, but their
conservative utility is still negative, the original recording check passes
0/102 views, and candidate coverage stays at 101/102. I do not retain this
variant as a replacement.
[Disagreement-aware training](outputs/publication_readiness_2026_09/moving_missingness_aligned_forecast_v1/readout.md).

I next trained a smaller direct motion model that weights past velocities.
Either new forecast gives the offline oracle feasible choices in all 102
views, up from 101, without changing the risk rules. The raw forecasters
still fail the risk check, and adding neighbors does not improve the paired
conservative score. The result motivates a new allocation test, not a
replacement model. [Direct kinematic forecasts](outputs/publication_readiness_2026_09/moving_direct_kinematic_forecast_v1/readout.md).

On that fixed bank, I compared independent, common-action and joint
selection with exactly the same per-query switch counts. The joint term
adds no clear relative-motion gain beyond its pair marginals. Filtering
before ranking recovers a small average gain over CV, but passes the
original risk test in only 2 of 102 views. I do not deploy either policy.
[Matched allocation](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/readout.md) |
[Budget-first follow-up](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/filter_first/readout.md).

I calibrated selected harm and reference estimates using held-source
predictions from other sources. Calibration reduced violations, but also
removed most useful interventions. Replacing one cost component at a time
on the original selections showed that harm underestimation was the larger
problem. [Selected-cost calibration](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/nested_calibration/readout.md).

I retrained the harm estimate on the selected region and held the switch
count fixed. Prediction error fell, but the original risk pass count did
not improve. Even with true outcomes, the screened candidate pool allows
only 30 of 102 views at those counts; the full bank has at least 72.
The next change is to preserve alternatives through allocation, not tune
the small-pool ranker. [Matched harm learning and oracle check](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/selected_positive_harm/readout.md).

I tested a chronological ledger that carries the two estimated budgets
forward without looking at later frames. At equal counts it improves
conservative utility over the same per-query optimizer. Adding learned
trajectories also beats the reoptimized physical bank on utility, but
reduces observed risk passes. [Causal ledger and physical control](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/readout.md).

I checked what funded those ledger choices. The queries predicted to earn
credit collectively overspent under the original accounting. Donor-based
discounting removed all carry but did not improve the risk result. A
standalone aggregate bundle reconstructs the ledger comparisons and risk
tables. [Credit and reproduction](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/credit_transport/readout.md).

I also tested whether forecast errors that are already observable tell me
when to switch again. With the same number of switches per frame, adding
agent and recording feedback does not improve utility reliably, and fewer
recording views pass the original risk test. Letting the model choose how
often to switch gives a small utility gain from agent feedback alone, but
still does not improve safety. I do not use this change for deployment.
[Past forecast feedback experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/resolved_feedback/readout.md).

I added smaller versions of each candidate forecast. At the same number
of switches, choosing among the expanded candidates improved conservative
utility, but none of the 102 recording views passed the original safety
test. Simply moving the original choices one quarter of the way away from
the baseline passed 29 views, at a cost in utility. This tells me the model
needs to get better at deciding where a smaller change is warranted.
[Bounded substitution experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/bounded_bank/readout.md).

Next I trained the model to estimate what one more increment of amplitude
would cost. It learned to make fewer large changes, but at the same final
amplitudes it did not allocate them more effectively. Even an oracle that
can see outcomes adds little by rearranging those amplitudes alone. I am
keeping this result as a limit on that approach, not a model upgrade.
[Incremental cost experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/bounded_bank/incremental/readout.md).

I then trained the ranker on differences between candidates in the same
frame, keeping the switch counts and predicted risk costs fixed. It chose
different agents, but gained less than the original ledger. The two new
rankers avoided large net losses on easy cases without controlling the
size of individual error increases. I keep the original risk test
and do not deploy either model.
[Within-query ranking experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/query_ranker/readout.md).

I also tried teaching the model from risk-constrained decisions. The
first comparison was uninformative: a teacher that already knew future
outcomes made the same choices with or without constraints. I replaced
that oracle's utility ranking with causal proposals fitted outside the
donor and evaluation sources. Risk screening changed over half of the
training demonstrations, but the student still did not improve on held
sources. Neither student meets the original risk requirements.
[Constrained demonstrations](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/feasible_teacher/nested/readout.md).

I then compared whole sets of replacements, with and without a term for
the two forecasts' relative motion. Neither improved the original
policy. More importantly, on this restricted candidate list even an
oracle could not satisfy the original risk rules in at least 75 of 102
recording views. I will work on which options the policy considers
before adding a larger model to score the same options.
[Whole-set comparison and support bound](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/feasible_teacher/setwise/readout.md).

I reopened the full candidate bank without changing the final risk
test. This reduced the number of recording views ruled out by the
fixed-budget diagnostic from 75 to 47, but did not increase actual
passes. For 28 views missing an easy reference, the original predicted
allowance blocked every option that could supply one, even though a
locally feasible option existed with observed outcomes. I now need to
improve how the model judges rejected options, not just rank more of them.
[Full-bank and reference-support experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/expanded_support/readout.md).

Refitting predicted risk for rejected candidates recovered some of that
utility at the same number of switches. It also hurt more easy cases,
and no extra recording view passed the full rule. Training only on
rejected candidates was not reliably better than training on all of
them. I keep this result as evidence of the gain and harm tradeoff,
not a deployment upgrade.
[Veto correction experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/veto_refit/readout.md).

I tested two ways to limit those reopened decisions, calibrating each
on sources that were separate from fitting and evaluation. Both cut
easy net losses relative to reopening without a check, but also lost
utility. The learned harm score did not reliably beat a simple limit
on how far the forecast moves from the baseline. Neither is ready to use.
[Release-guard comparison](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/veto_refit/guard/readout.md).

I then matched calibration recordings by motion in the first observed
query. The matched recordings were closer, but the thresholds did not
transfer better than selecting the same number without looking at
motion. The full risk pass count did not improve. I keep this result
as a limit of the first-query features, not a reason to pick the
farthest recordings instead.
[Context-matched calibration](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/veto_refit/guard/context/readout.md).

I also held the fitted model and switch count fixed while doubling the
number of calibration sources from three to six. The extra sources did
not improve utility on held sources or increase the number of views
that passed the original rule. More calibration alone did not fix this
policy. [Calibration size comparison](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/veto_refit/guard/size/readout.md).

I also changed the learned risk head itself. A forecast that looks less
risky only when a particular training source is present now gets a smaller
risk reduction. At the same switch count, this reduced easy-case violations
but lost prediction utility. I did not adopt it for deployment.
[Source-stable risk head](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/veto_refit/guard/stability/readout.md).

I then kept the risk head fixed and changed the gain model. It scores
whether each candidate moves in a direction that would reduce the
baseline's error. It changed many choices, but its small utility gain
was not reliable, and only four of 102 views passed the original rule.
[Directional gain experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/directional_utility/readout.md).

For the next external study I checked which recordings could have fitted
these two cost heads. All 34 are covered by the earlier cross-archive image
screen. This checks training provenance, not external performance; I have
not selected an external policy or scored the new archive.
[Recording coverage](outputs/publication_readiness_2026_09/dronecrowd_controller_coverage_v1/README.md).

To see whether the forecasts themselves were the main limitation, I
computed what could be selected with hindsight under the original
risk rules. The bank contains useful, admissible combinations in 101
of 102 development views when the timing and number of switches may
change. Fixing the old switch counts and applying the predicted budgets
leaves far fewer. This points to the allocation and cost estimates,
but does not show that the right actions are identifiable online.
[Hindsight capacity](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/prefix_capacity/readout.md).

I then kept each recording's total number of interventions fixed and
allowed the timing to move. Even with predicted prefix budgets, this
found more useful hindsight choices. Simply carrying unused slots
changed no online decisions, and a rule that deliberately waited lost
some utility. I keep the timing result as a diagnostic, not a policy
upgrade. [Timing comparison](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/timing_capacity/readout.md).

I also tested whether a learned easy-support score could open recordings
where the direct policy never switches. Neither calibrated score did.
A hindsight check separates missing candidates from missed choices:
nine of the 18 zero-switch views have a useful first-query option,
six only have one later, and three have none in this bank.
[Opening action experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/easy_joint_anchor/readout.md).

I then trained a controller to compare switching now with waiting one
eligible query. It did not improve the original policy. The ranking
diagnostic explains a limit: a useful opening exists in the first two
queries for 15 of the 18 zero-switch views, but no tested ranker puts it
first. I will improve candidate selection before expanding the waiting
rule. [Stop versus wait](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/opening_stop_wait/readout.md).

I trained two useful-action rankers to test that bottleneck directly.
The pairwise loss improved shortlist recall but did not improve the
calibrated policy. At the same eight-slot budget, a simple shortlist
that retains different forecast families also found more useful
choices. [Ranking and shortlist comparison](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/opening_pair_rank/readout.md).

I then fixed that family-diverse shortlist and compared final choice by
predicted gain with choice after a predicted risk penalty. Neither
calibrated policy added a full pass. The shortlist contained useful
choices that neither selector picked, so I treat candidate availability
and action choice as separate results.
[Fixed shortlist experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/shortlist_cost/readout.md).

I also tested whether the model could use errors it had already seen.
None of the first two eligible queries in the 18 no-switch views had
12-step feedback available for a shortlisted action. Short forecasts
checked against the eight observed positions helped slightly at equal
switch counts, but did not add a pass under the original 2% rule.
[Past-error experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/shortlist_history/readout.md).

I tested neighbor motion and predicted path proximity on the same
shortlist. Neither helped the policy pass more recordings, and both
added easy-case violations at the same switch count. I kept these
features out of the current policy.
[Interaction experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/shortlist_interaction/readout.md).

I also tried spending the same eight slots on different gain and
risk rankings. That found two opportunities the family-diverse list
missed but lost five it had kept. I kept the family-diverse list:
neither new choice rule improved complete policy passes.
[Candidate retrieval experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/shortlist_portfolio/readout.md).

I replayed those five losses. The cost ranking displaced two
forecast families that supplied the missed actions. Keeping six
family slots restored three of the losses but still found fewer
useful openings than the original list. With the same scoring head
it added no calibrated intervention.
[Retrieval loss audit](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/shortlist_exclusions/readout.md).

I kept all forecast families and learned a new ranking within each
one. It found six useful early winners across the recordings, but
adding those actions to their original recordings still produced no
complete pass. The new ranking also increased easy-case violations,
so I did not keep it in the policy.
[Within-family ranking experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/family_supported_gain/readout.md).

I also tested removing existing switches instead of adding an early
one. With hindsight, 30 of 102 recording views admit a subset that
passes the original rules without losing paired utility. The learned
rule could not find a calibration threshold that preserved both
utility and risk. It made no deletions.
[Recording pruning experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/record_pruning/readout.md).

Training on verified removal sets did make the model intervene, but
it still repaired no held-out recording. One version removed the
easy-reference support from a recording that previously passed.
I kept the original policy.
[Removal-set supervision](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/pruning_teacher/readout.md).

I tested whether the model could recognize a repairable recording
before deciding which actions to remove. The score was weak, and
neither routing nor keeping a reserve of predicted easy support
added a complete pass. The result does not support using the new
router in the policy.
[Recording routing experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/pruning_router/readout.md).

To see whether deletion was the wrong restriction, I measured the
hindsight capacity of adding one support action and then removing
harmful actions. The full bank can repair 33 of 35 views with
undefined easy support; the old eight-action list allows only 18.
This identifies a retrieval problem, not a deployable result.
[Support acquisition experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/support_acquisition/readout.md).

The support-specific list retained 20 rather than 18 feasible
views at the same eight-action budget. The operating rule did
not improve: its proposed insertions in missing-support donor
records supplied no observed easy reference, so no threshold
qualified. I kept the original policy.
[Support retrieval and joint repair](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/support_joint/readout.md).

I trained an agent-level easy-reference score separately from
the edit decision. The boosted model ranked better on average,
but did not beat the stronger control reliably and did not
improve its first-query hit rate. I did not add it to the policy.
[Easy-reference ranking experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/easy_reference_rank/readout.md).

I then trained two identical query rankers, emphasizing the first
two queries in only one of them. The extra weight made the first
choice worse: 20.56% versus 31.90% without it. I did not carry the
early-weighted model into the policy.
[Early-query ranking experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/early_reference_listwise/readout.md).

In the [working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v10/manuscript.md),
I separate what can be repaired after seeing the labels from what the
model can actually do. Joint edits could recover 33 of 35 missing-reference
views in hindsight, but only 18 fit the original shortlist. A new shortlist
admitted 20; the past-only policy still found no change that qualified.
I have not changed the deployed policy or the evaluation rules.

I then trained one network to estimate easy-reference support and
conditional gain separately. It picked a non-losing action with
observed reference in 9 of 102 first queries, versus 3 for direct
value prediction. The gain stopped there: neither method recovered
a whole recording under the original rules.
[Support and value experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/reference_hurdle/readout.md).

I trained on whole-recording outcomes next. That recovered one
missing reference, but the recording still failed the original
risk check. In hindsight, keeping the insertion and withdrawing
one existing switch would have passed. The learned policy did not
make that joint decision, so I have not changed the operating model.
[Whole-record supervision experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/record_rescue/readout.md).

I also tested a same-query exchange that keeps the number of
switched agents fixed. Neither learned score recovered a whole
recording. The action set itself is limited: five of 35 missing
references could be repaired in hindsight, or eight when the
withdrawal could occur in a later query at the same recording count.
[Count-preserving exchange experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/atomic_swap/readout.md).

I let the model insert a forecast first and withdraw an original
switch only when warranted. It made four insertions but never
withdrew. None of those four was in a recording that lacked easy
reference, so the experiment did not establish a withdrawal benefit.
[Optional withdrawal experiment](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/optional_withdrawal/readout.md).

I added geometry from the eight observed head boxes to test
whether overlap and apparent size improve insertion retrieval.
The two feasible first-query choices came from the same recording,
and neither was taken by the calibrated policy.
[Box geometry comparison](outputs/publication_readiness_2026_09/moving_record_joint_actions_v1/causal_ledger/box_visibility/readout.md) ·
[Insertion appendix](outputs/publication_readiness_2026_09/evidence_manuscript_v10/insertion_appendix.md).

I have now streamed all 112 DroneCrowd recordings to check that
the eight-step input path works before an external test. This does
not open confirmation labels or assign roles. The two proposed
splits and the checkpoint source checks are recorded in the
[external input preflight](outputs/publication_readiness_2026_09/dronecrowd_blind_preflight_v1/README.md).

I tested the external scoring code on the existing development
recordings. Changing the reference cuts changed pass counts even
though the decisions and overall utility were identical. The
[scoring rehearsal](outputs/publication_readiness_2026_09/dronecrowd_scoring_rehearsal_v1/README.md)
keeps the original result and shows why the cuts must be fixed
before confirmation.

I kept the same switches and reduced how far their forecasts
departed from constant velocity. At one quarter of the original
amplitude, 15 rather than 2 recording views passed, but the same
35 still lacked easy reference.
[Amplitude control](outputs/publication_readiness_2026_09/moving_frozen_amplitude_v1/readout.md).

### Earlier Viewpoint Study

I tested whether explicit viewpoint conditioning helps the trajectory network.
Past-image motion aligns observed agent and neighbor histories to the current
view. A fixed extrapolation restores predicted image coordinates. The network,
loss, source exclusions and 4,000-update budget stay matched to the original.

**This model repair fails its registered component comparison.** The new actor
loses **5.133 percentage points** against the original on known-label whole-
12-step ADE gain. Its paired completion lower is **-21.447pp**, with an
exploratory source-bootstrap interval of **[-33.682, -9.911]**. Only 3 of 14
sources improve, and both producer-fold means are negative.

| Whole-12 Diagnostic | Known-Label ADE Gain vs CV |
|---|---:|
| Original trajectory actor | +15.378% |
| Camera-conditioned actor | +10.245% |
| Camera-compensated CV | +2.828% |

These are source-balanced exposed-development diagnostics, not deployment
scores. A camera-CV benefit does not establish added neural value. The new raw
actor exceeds 2% known-easy positive harm in 33 of 34 recordings; this is not
a complete selected-policy certificate. All six matched final training batches
also have higher loss after conditioning. That does not identify whether past
alignment or future camera extrapolation causes the failure: they were tested
together. **No new controller or deployment follows.**

All six treatment models completed 4,000 updates. A full original-control
refit, one full fixed-design treatment refit, and all twelve actor/camera-CV
prediction banks and scalar results reproduce exactly. A tool interruption
was recovered from its unchanged checkpoint; no budget was reduced. Runtime
reproduction is not independent scientific confirmation.

The study retains all 125,540 previously exposed donor windows, 14 source
groups and 34 recordings. It observes 8 steps and predicts 12 at native stride
1 in image coordinates. I kept the original 2% and support/undefined-reference
rules, without identity-fallback certification or a new own-B/C/H readout.
It does not establish true 3D, a foundation model, verified meters or seconds,
or submission readiness.

[Complete camera experiment](outputs/publication_readiness_2026_09/moving_camera_frame_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_camera_frame_v1/failure_analysis.md),
[reproduction instructions](outputs/publication_readiness_2026_09/moving_camera_frame_v1/operation_zh.md).

The earlier past-image feature branch also failed its matched comparison:
known difference -1.804pp and paired lower -10.931pp [-17.826, -5.288].
Its nonzero learned weights and unclipped worst source rule out a completely
inactive branch or input clipping alone as sufficient explanations.
[Earlier actor study](outputs/publication_readiness_2026_09/moving_visual_actor_v1/README.md).
The [English working manuscript](outputs/publication_readiness_2026_09/evidence_manuscript_v22/README.md)
keeps these native-stride results separate from older European stride-12 and
SDD studies. It remains an evidence draft, not a submission-ready paper.
The [companion supplement](outputs/publication_readiness_2026_09/evidence_manuscript_v22/README.md)
explains how I distinguish observed utility, missing-label bounds and
recording-level failures, with instructions to rebuild the tables.

A one-sided operator ablation was stopped before outcome scoring after a
synthetic counterexample exposed information loss: aligned history alone can
be identical under different camera motions while native future labels differ.
Its recovery pilot and full original control passed, but only the first new
model reached 2,160 updates. I do not report it as a completed experiment.
This is why the current replacement retains both native and aligned history.
[Design diagnosis](outputs/publication_readiness_2026_09/moving_camera_factor_v1/design_stop.md).

The [evaluation supplement](outputs/publication_readiness_2026_09/evidence_manuscript_v10/supplementary_math.md)
now makes an important distinction explicit: available-label error and the
registered row-support completion envelope are not complete-trajectory ADE.
The derivations and synthetic checks explain the existing quantities without
changing the evaluation contract or any historical score.

A [standalone aggregate-evidence package](outputs/publication_readiness_2026_09/native_actor_reproduction_v1/README.md)
replays the three earlier completed actor studies in a clean directory without
Torch,network access or the project workspace. It preserves their negative
results and distinct bootstrap seeds. This is reporting reproducibility,
not independent validation or a complete model-training release.

### Previous Controller Experiment

I tested a new observation from the past images: native image bounds and
background-motion context. All 24 matched neural cost heads completed 4,096
updates, and 24 complete from-scratch refits reproduced exactly. Decisions
were frozen before scoring all 62,466 existing development windows.

| Inputs | Available-Label ADE Gain vs CV | Original Recording Checks Passed | Worst Signed Easy Loss |
|---|---:|---:|---:|
| No image | +0.700% | 18/72 | 12.41% |
| Image bounds | +1.261% | 26/72 | 9.55% |
| Quality / geometry control | +5.290% | 12/72 | 47.35% |
| Background motion | +5.867% | 12/72 | 118.26% |

The registered motion-minus-quality paired lower contrast is **+0.462pp
[-1.095, +2.294]**. Its interval crosses zero, and the motion model still
harms easy cases. Better training fit and better prediction of label availability
do not produce accurate harm estimates. **The component screen and original
safety contract fail; the new model is not deployed.** All predictions and
reported scalars reproduce exactly in a separate replay.

These are exposed-development results, not an independent test. The72views
repeat24recordings across three risk-head seeds, not three forecasters.
The original2%/support/NA rules and previous failures remain unchanged.
The task is native observation8/prediction12 in image coordinates, not SDDt50
or a calibrated physical world. The project is **not submission-ready**.
[Complete comparison](outputs/publication_readiness_2026_09/moving_visual_risk_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_visual_risk_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_visual_risk_v1/operation_zh.md).

### Previous Actor Repair

I retrained the actual trajectory predictor to test whether better early-step
supervision repairs the unsafe forecasts. All six matched neural models finished
4,000 updates, and a full old-model control reproduced exactly. The new loss
helps early predictions but does not improve the whole forecast consistently:

| Forecast Prefix | Gain Difference vs Old Predictor | Improving Sources |
|---|---:|---:|
| 1 step | +12.439pp | 12/14 |
| 3 steps | +2.915pp | 12/14 |
| 6 steps | +0.677pp | 7/14 |
| 12 steps | +0.098pp | 6/14 |

These are equal-source, query-balanced differences on 125,540 existing training-
donor windows, not a new held-out test. The whole-forecast conservative lower
contrast is **-9.474pp [-17.809, -3.083]**; this bounds unknown-label uncertainty,
not the actual unobserved error. One producer-fold mean is also negative before
that adjustment. All 34 raw neural recording diagnostics still exceed the
original 2% known-easy positive-harm reference. **The component screen fails;
the new model is not deployed.** No rule, support requirement or test role was
changed to make the result pass. The project remains **not submission-ready**.
All six new prediction banks and every reported scalar now reproduce exactly
in a separate replay, without further training.
[Complete model experiment](outputs/publication_readiness_2026_09/moving_prefix_actor_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_prefix_actor_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_prefix_actor_v1/operation_zh.md).

The completed image experiment above uses 1,851 past training-source images;
1,694/1,817 transitions meet the fixed measurement rule, and all125,540causal
feature rows reproduce exactly. The full exposed development input roster is
also retained, including failed measurements. This is a conventional image-motion
proxy, not calibrated camera geometry. Future target arrays stay unopened during
measurement and inference. Image availability is separate from predictive lift.
[Measurement protocol](outputs/publication_readiness_2026_09/moving_past_background_v1/protocol.md).

### Previous Support / Context Study

I tested whether more source-excluded training support makes global past-motion
context useful to the controller. All 96 Torch cost/support models and 96 complete
exact refits are finished. The four cells retain the original risk rules:

| Training Support / Features | Available-Label ADE Gain vs CV | Original Checks Passed | Worst Recording Easy Loss |
|---|---:|---:|---:|
| Own / local | +6.805% | 3/72 | 129.66% |
| Own / global | -0.313% | 4/72 | 156.26% |
| Augmented / local | +0.668% | 23/72 | 22.12% |
| Augmented / global | +1.038% | 13/72 | 19.86% |

**Lower training loss did not produce a safe policy.** All 48 paired context
models fit better, but the registered augmented global-minus-local paired
completion-lower contrast is only **+0.199pp [-0.820, +1.254]**. The interval
crosses zero. The new global model also loses support and fails the unchanged
recording-level 2% requirements, so I am not deploying it.

These are 62,466 exposed MovingDroneCrowd++ agent-window rows, native observation8 /
prediction12 in image coordinates. The 72 views repeat 24 recordings over three
risk-head seeds, not three forecasters. No independent confirmation or new
trajectory training is claimed. The project remains **not submission-ready**.
[Complete experiment](outputs/publication_readiness_2026_09/moving_supported_global_motion_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_supported_global_motion_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_supported_global_motion_v1/operation_zh.md).

The matched simple controls also remain important: guarded fixed1/32 gains
1.434% on the same cohort but passes only 5/24 recording checks, with worst
signed easy loss8.95%. All six original amplitudes are reported; no coefficient
is selected after scoring. Neither mean accuracy nor empty fallback support
is treated as a safety pass.
[Full control matrix](outputs/publication_readiness_2026_09/moving_simple_controls_v1/README.md).

I completed the prefix audit on all 125,540 permitted training-side agent
windows and reproduced every scalar in a separate replay. Nine of 14 sources
have worse first-step forecasts even though ten improve over the available
12-step average. The first-step direction cue is already recoverable from
the existing inputs, and opposing motion explains little of the total harm.
This points toward testing the forecast training loss, not adding a duplicate
feature or another threshold. No evaluation rule or deployment has changed.
[Training-side audit](outputs/publication_readiness_2026_09/moving_fitting_prefix_audit_v1/README.md).

The follow-up prefix-balanced producer experiment is now complete and negative
on its whole-forecast screen, as summarized above. It changes actual neural
training, not the original risk contract or a held-out evaluation role.

### Previous Reference-Context Result

I tested whether sharing the frozen expert's causal rollout context could repair
reference-cost learning without changing harm or label-support predictions.
All 48 matched reference models and 48 complete exact refits are finished.

**Total reference prediction improves, but easy-case protection is still unresolved.**

Shared context improves total-reference MSE by about 6.1% over its matched
history-only refit, but worsens easy-reference MSE by 21.6%. Its available-label
ADE gain over causal CV is 0.791%, versus 0.667% for the frozen prior. However,
the registered paired completion-lower contrast is only +0.133 percentage points
with an exploratory source interval of [-0.141, +0.556], so stable added utility
is not established.

Only **24/72 required repeated recording views** pass the original contract.
There are 26 known-easy harm violations; worst signed easy-net degradation is
**9.02%**, above the unchanged 2% limit. A real failure case also shows the frozen
harm head underestimating loss. Shared easy-support, convex amplitude constraints
and global past-motion context also have prior negative results. I am testing
whether data support changes the value of causal global motion, rather than
presenting those old ideas as new or loosening evaluation. No deployment change
follows this result.

These remain MovingDroneCrowd++ exposed-development, native observation-8 /
prediction-12 image-coordinate results, not independent confirmation or a new
trajectory dynamics result. The 72 views repeat 24 recordings over three risk
seeds; the forecaster has one fixed seed. The project is not submission-ready.
[Complete comparison](outputs/publication_readiness_2026_09/moving_expert_reference_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_expert_reference_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_expert_reference_v1/operation_zh.md).

### Previous Matched Cost Result

Separating cost learning from label-support learning improves trajectory utility
in the latest matched experiment. I trained 48 Torch heads and reproduced all
48 complete fits exactly, then evaluated fixed independent and joint choices
over the original forecast-amplitude bank. Both losses use the same architecture,
initialization, source-excluded training draws and budget.

**The model improves utility, but the complete strategy is still not safe to deploy.**

The registered paired completion-lower gain over the matched control is
**+0.445 percentage points [95% exploratory source interval: +0.091, +0.931]**.
All three risk-head seeds have positive mean contrasts; two of eight source
lower contrasts remain negative. The forecast teacher is still a single seed.

The new joint policy gains 0.667% in available-label ADE over causal CV, but
only **23/72 required recording views** pass the original risk/support contract.
There are 27 known-easy harm violations. Joint selection exposes optimistic
reference and harm predictions: utility rises while safety worsens relative
to the new independent policy. This is the next model problem, not a reason
to loosen the 2% rule or certify empty support as a pass.

These are MovingDroneCrowd++ native observation-8/prediction-12 development
results, not an SDD t50 result or independent confirmation. The 72 views repeat
24 recordings over three risk seeds; uncertainty uses eight source blocks.
All 62,466 rows and 374,796 legal candidates remain. Deployment is unchanged,
and the project is not submission-ready.
[Complete comparison](outputs/publication_readiness_2026_09/moving_role_separated_cost_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_role_separated_cost_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_role_separated_cost_v1/operation_zh.md).

### Previous Complete-Risk Training Result

I tested a concrete training repair after the complete-policy audit: add the
conservative unknown-label contributions omitted by donor fitting. All 24 new
Torch heads and 24 full exact refits are complete, with the same model, causal
features, source exclusions, training budget and original 2% rules.

**Adding unknown-risk targets alone does not repair safe intervention.**

The mean rule admits fewer unknown tracks (81 to 68), but more recording views
violate known-easy harm (15 to 19). Only 1/24 views passes the complete contract;
unanimity drops from 2/24 to 1/24. Observed-risk target error worsens in every
source/seed comparison while bound-target error mostly improves. The next
model repair will separate these competing targets and model reference support,
not loosen the rule or claim an empty fallback has passed. Deployment stays
unchanged, and the project is not submission-ready.
[Model comparison](outputs/publication_readiness_2026_09/moving_complete_risk_donors_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_complete_risk_donors_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_complete_risk_donors_v1/operation_zh.md).

### Initial Complete-Policy Audit

I reconstructed all 62,466
past-eligible rows in eight exposed development sources, including unknown
labels and non-interventions, and kept the original per-record 2% risk and
support requirements. No new held test, training or threshold search was used.

**Trajectory accuracy improves, but neither consensus policy is safe to deploy.**

The fixed mean rule improves available-label source-macro ADE by 1.313%, and
unanimity by 0.336%, versus the training-defined motion floor. Conservative
completion lower gains are 0.824% and 0.306%. Yet only 1/24 and 2/24 required
recording views pass the full original risk/support contract. This is not just
an empty-denominator problem: there are 15 and 9 known-easy positive-harm
violations, respectively. Unknown labels add further conservative risk.

All recordings have easy observations before selection, but consensus removes
much of the selected easy reference mass. The next repair targets complete-risk
training and reference support, not a looser safety rule. Empty support stays
not estimable; there is no identity-fallback exemption. Deployment is unchanged.
[Complete results](outputs/publication_readiness_2026_09/moving_complete_donor_policy_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_complete_donor_policy_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_complete_donor_policy_v1/operation_zh.md).

### Risk-Head Seed Replication

The donor benefit now replicates across risk-head seeds 17/29/43 with the
forecast teacher fixed. I trained 48 additional heads and repeated every new
fit exactly, retaining the verified seed17 models rather than selecting a
favorable run. The information criterion passes in 3/3, 2/3 and 2/3 contexts.

**The component learns better; the subsequent complete-policy check fails.**

Average individual-head decision cost falls 12.16% versus matched own-only
training, with an exploratory source-component interval of [2.66%, 27.18%].
The forest comparison is 4.15% [-4.75%, 8.12%], which does not establish a
reliable advantage over that stronger control. One seed/context has worse MSE.
These intervals use eight exposed source components, not overlapping windows,
and do not describe the separately aggregated score ensembles.

The fixed three-head unanimity rule reduces harmful known-proposal admissions
to 7/26/26, versus 46/229/602 for the original single head and 0/1/0 for the
forest. It also rejects useful predictions. These component counts exclude the
unknown-label contributions and reference/support conditions evaluated in the
new complete-policy result above. The outer test stays closed. Deployment and
the original rules remain unchanged.

These are three risk-head seeds, not three forecaster seeds or independent
world-model confirmation. The project is not submission-ready.
[Replication and uncertainty](outputs/publication_readiness_2026_09/moving_donor_seeds_v1/README.md),
[fixed consensus](outputs/publication_readiness_2026_09/moving_donor_seeds_v1/consensus/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_donor_seeds_v1/failure_analysis.md).

### Previous Seed17 Donor Comparison

The new source-held donors improve neural risk learning in the matched
development comparison. I trained 24 heads and repeated all 24 fits exactly
before evaluation. Each receives the same 4,096-update budget and original
causal features; the comparison is own-B only, producer-A only, and a 50/50 mixture.

**The mixture improves information, but is not yet safe to deploy.**

Decision cost falls 20.71%, 32.27% and 10.69% versus matched own-only training;
risk MSE also falls in all three contexts. All eight source folds have lower
total decision cost. This supports the donor treatment rather than another
blind increase in training time. It does not isolate source count from teacher
size and candidate-mixture effects, and remains a one-seed development result.

Against the more conservative forest, total cost improves in two contexts but
worsens in one, and false-safe cost is higher in all three. Source060 also has
more harmful admissions than own-only despite its lower total cost. The 3/3
information screen is not a safety-gate pass. Deployment, the original 2% rules,
empty-support NA and the closed outer test remain unchanged.
[Matched results](outputs/publication_readiness_2026_09/moving_donor_augmentation_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_donor_augmentation_v1/failure_analysis.md),
[reproduction](outputs/publication_readiness_2026_09/moving_donor_augmentation_v1/operation_zh.md).

### Source-Held Forecast Donor Bank

I trained six new EqMotion controls to expand risk-learning source coverage
without relaxing the safety rule. Each completed 4,000 updates. Their forecasts
are held out by source inside the original predictor training sets, keeping
outer calibration/evaluation roles closed. The bank now covers 125,540 rows
from 6 and 8 sources in the two separate parent training sets.

**The training-data repair is complete; safer intervention is not yet proved.**

All six forecast caches replay exactly. Donor ADE beats causal CV on 10 of 14
sources and loses on four; all sources and unknown-label rows remain in the
record. These are training-donor diagnostics, not a new independent test or a
strongest-baseline result. The matched donor comparison and its risk-head seed
replication are recorded above. Deployment and the original 2% rules
remain unchanged; the paper is not submission-ready.
[Training and donor results](outputs/publication_readiness_2026_09/moving_producer_crossfit_v1/README.md),
[source eligibility](outputs/publication_readiness_2026_09/moving_producer_crossfit_v1/eligibility_report.md),
[reproduction](outputs/publication_readiness_2026_09/moving_producer_crossfit_v1/operation_zh.md).

### Previous Budget-Controlled Risk Experiment

I tested whether the neural risk failure was simply due to too little training.
I continued all eight saved heads, adding 32,871 updates with the same model,
features, loss, sources and original risk rules. Only the fixed final checkpoint
was evaluated after every fit and exact repeated continuation had completed.

**More training improved fitting, but did not repair transfer.**

Fitting loss fell in all eight folds; source-held risk MSE worsened in six.
The decision-cost reductions versus the forest control are -4.72%, -68.65%
and -6.69% across the three development contexts. The screen passes 0/3, so
deployment is unchanged. These are risk-model results, not trajectory gains.

The fitting banks each contain only one or two source components. I am now
prioritizing compatible source coverage and conditional risk learning over
another unstructured increase in training time. The original 2% and support
requirements remain; empty support is NA. No new reserved test was opened.
The paper is not submission-ready.
[Results and loss traces](outputs/publication_readiness_2026_09/moving_risk_convergence_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_risk_convergence_v1/failure_analysis.md),
[source coverage](outputs/publication_readiness_2026_09/moving_risk_convergence_v1/source_coverage.md).

### Previous Neural Risk Result

I trained a small neural risk model to adjust the mismatch between a query
and its reference histories. Plain neural prediction and a linear correction
serve as controls; the reference set, causal inputs and original rules stay fixed.

**The networks trained and reproduced, but did not improve safe intervention.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | -42.675% | -6.914% | 0 to 94 |
| 1 | -18.759% | -54.522% | 1 to 353 |
| 2 | +32.192% | -5.517% | 0 to 1288 |

Eight Torch networks and eight ridge controls complete; every fitting row gets
32 epochs. All models reproduce by full refitting. Correcting reference risk
helps modestly versus the plain network, but not versus the forest control.
The information screen passes 0/3 contexts. Actual training losses are retained, and
convergence is not claimed. This is risk-model development, not new neural
trajectory success, independent confirmation or a safety certificate.
The original 2% and support requirements remain, empty references stay NA,
and deployment is unchanged. The paper is not submission-ready.
[Results and loss traces](outputs/publication_readiness_2026_09/moving_neural_transport_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_neural_transport_v1/failure_analysis.md),
[Chinese reproduction guide](outputs/publication_readiness_2026_09/moving_neural_transport_v1/operation_zh.md).

### Previous History-Matching Result

I tested whether similar causal histories from other sources can identify
useful interventions that risk forests miss. The new local risk model uses
32 source/agent buckets per query and excludes the query's entire source.
It leaves the features, targets and original safety rules unchanged.

**Matching history carries information, but still does not make intervention safe.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | -0.427% | -4.482% | 0 to 107 |
| 1 | -0.199% | -121.691% | 1 to 723 |
| 2 | +33.420% | -2.568% | 0 to 521 |

Correct matching beats a fixed misaligned control, but loses to the stronger
forest on decision cost in all three settings. The fixed information screen
fails 0/3. All 48,061 known proposals were processed, and all eight complete
retrievals reproduce exactly. These are exposed development component results,
not trajectory gains, independent test evidence or a safety certificate.
Deployment stays unchanged. The original 2% and support requirements remain;
empty reference support is not estimable, not passed. No identity exemption.
The paper is not submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_track_analogue_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_track_analogue_v1/failure_analysis.md),
[Chinese reproduction guide](outputs/publication_readiness_2026_09/moving_track_analogue_v1/operation_zh.md).

### Previous Channel-Head Result

I traced why recent risk-regression improvements did not improve intervention
decisions. Almost all of the error reduction occurred on proposals that both
models still rejected. I then tested a concrete model change: separate
all-agent and easy-agent risk regressors, keeping the total tree budget,
causal features, targets and safety rules unchanged.

**Separating the heads did not repair safe intervention.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | +2.909% | -0.034% | 0 to 0 |
| 1 | +0.372% | +0.461% | 1 to 19 |
| 2 | -3.898% | 0.000% | 0 to 0 |

The fixed information screen fails 0/3. Sixteen new model objects and eight
verified controls reproduce by full refit; that establishes reproducibility,
not effectiveness. These are exposed source-held component diagnostics, not
trajectory improvements or an independent test. I keep the current deployment
unchanged and retain the original 2% risk and support requirements. Empty
reference support remains not estimable. The paper is not submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_channel_specialist_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_channel_specialist_v1/failure_analysis.md),
[Chinese reproduction guide](outputs/publication_readiness_2026_09/moving_channel_specialist_v1/operation_zh.md).

### Previous Motion-Frame Result

I completed a motion-frame repair suggested by the tail audit: express motion
in the agent's own history scale and retain the original scale context.
Sixteen new models and eight verified controls keep the same signed targets,
risk requirements and data roles. All models reproduce by full refit.

**Better risk regression still does not establish better safe decisions.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | +9.825% | +0.115% | 0 to 0 |
| 1 | +0.709% | +0.054% | 1 to 6 |
| 2 | +30.216% | 0.000% | 0 to 0 |

The fixed information screen fails 0/3. In context 2, all 38,579 known
proposals remain rejected. These are source-held risk-model diagnostics,
not trajectory improvements, independent confirmation or safety certificates.
No model is promoted. The original 2% and support rules remain unchanged;
empty controls stay not estimable. The paper is not submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_ego_frame_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_ego_frame_v1/failure_analysis.md),
[Chinese summary of both repairs](outputs/publication_readiness_2026_09/moving_ego_frame_v1/model_repair_summary_zh.md).

### Completed Forecast-Tail Test

I trained eight positive-cost risk probes to trace the harmful tail of the
frozen forecast bank, against eight verified signed-risk controls. All models
reproduce by full refit. The original safety rules and reserved tests stay unchanged.

**Harm is concentrated, but causal tail identification still transfers poorly.**
At a fixed10% ranking weight, easy-cost capture changes from
19.20/3.42/38.23% to20.67/4.08/39.10% across three contexts. The internal
information screen fails0/3. These are diagnostic cost-capture rates, not
trajectory gains or safety passes. Fitting-source capture is much higher than
excluded-source capture. The next controlled repair concerns the motion
representation's dependence on neighbor scale, not another threshold sweep.
No new model is deployed, and the paper is not submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_forecast_tail_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_forecast_tail_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_forecast_tail_v1/protocol.md).

### Previous Cost-Partition Result

I tested a targeted change to risk learning: fit partitions around useful and
costly decisions, while keeping the original expected-risk readout. Sixteen new
models and eight verified controls use the same inputs and capacity limits.
All models reproduce by full refit; the safety rules and decision threshold
are unchanged.

**Substantially better risk regression still did not yield a safe decision upgrade.**

| Context | Risk-MSE reduction | Decision-cost reduction | Harmful admissions, control to new |
|---|---:|---:|---:|
| 0 | +10.744% | -1.830% | 0 to 6 |
| 1 | +1.042% | +0.041% | 1 to 9 |
| 2 | +35.804% | +0.284% | 0 to 3 |

Positive reductions mean better. The fixed internal follow-up screen fails 0/3.
These are source-held fitting diagnostics, not trajectory gains or independent
confirmation. I do not promote this model or open another reserved-test readout.
The next question concerns the harmful tail of the candidate forecasts: whether
the predictor itself needs repair, rather than another selector-weight sweep.
The paper is not yet submission-ready.
[Results](outputs/publication_readiness_2026_09/moving_cost_partition_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_cost_partition_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_cost_partition_v1/protocol.md).

### Last Completed Information Test

I tested whether richer directional information about nearby agents improves
intervention decisions. The comparison adds step-by-step candidate-to-peer
geometry, including a deliberately mismatched association control, and leaves
one fitting source out at a time. All 48 fixed linear and nonlinear fits are
complete and reproduce exactly.

**Better risk prediction did not produce reliable decision gains.**

| Context | Nonlinear risk-MSE reduction | Decision-cost reduction | Outcome |
|---|---:|---:|---|
| 0 | -1.262% | -0.000449% | Worse prediction |
| 1 | +0.997% | +0.189098% | More harmful admissions |
| 2 | +5.000% | 0.000000% | All candidates still rejected |

Positive reductions mean better. The primary follow-up screen passes only 1/3,
below the registered requirement of two. In context2, the favorable error change
does not alter any of the 38,579 known candidate decisions. Context1 increases
harmful admissions from 1 to 10. I therefore do not expand this feature branch
into a new neural model or promote it for deployment.

This is an exposed-development information test, not independent confirmation
or a trajectory-dynamics improvement. Original 2% risk and support rules remain
unchanged; there is no identity-fallback exemption or new reserved-test readout.
The next question is why the risk estimate fails to distinguish useful switches
from costly mistakes, with earlier negative cost-head experiments retained.
The paper is not yet submission-ready.
[Complete results](outputs/publication_readiness_2026_09/moving_peer_path_probe_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_peer_path_probe_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_peer_path_probe_v1/protocol.md).

### Last Completed Model Experiment

I separated two model-input choices that the previous experiment changed
together: which forecast represents a peer, and whether fallback peers remain
visible. Six new neural fits and six verified existing controls complete the
four-cell comparison, with matched capacity, initialization, training and loss.

**Keeping fallback peers did not produce a reliable decision improvement.**

| Context | Risk MSE reduction | Decision-cost reduction | Harmful-admission-cost reduction |
|---|---:|---:|---:|
| 0 | +0.043% | -0.043% | +8.922% |
| 1 | -0.279% | -4.108% | -19.095% |
| 2 | -0.402% | +0.675% | +9.027% |

Positive reductions mean better. These are fitting diagnostics, not test
trajectory gains. Retaining fallback peers partially repairs two prior failures,
but the registered primary fails the fixed internal screen in all three contexts.
No additional development or reserved-test evaluation is opened.

All six fresh fits finish 1,000 updates. Resume, legacy-engine replay, a full
repeat and all twelve cells' metrics agree exactly. The original 2% safety and
support rules remain unchanged; no identity exemption or deployment upgrade is
introduced. The next question is whether candidate-relative peer information
adds source-held-out predictive value beyond the existing pooled representation.
The paper is not yet submission-ready.
[Results and loss curves](outputs/publication_readiness_2026_09/moving_peer_participation_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_peer_participation_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_peer_participation_v1/protocol.md).

### Last Completed Peer-Context Experiment

I completed six matched neural fits to test whether risk estimation benefits
from seeing each peer's actual proposed forecast, rather than a same-index
hypothetical forecast. Both models have the same capacity, initial predictions,
training budget and loss. The original 2% risk rules remain unchanged.

**Proposal-conditioned context did not consistently improve decisions.**

| Context | Risk MSE reduction | Decision-cost reduction | Harmful-admission-cost reduction |
|---|---:|---:|---:|
| 0 | +0.067% | +0.920% | 0.000% |
| 1 | +1.078% | -6.975% | -29.496% |
| 2 | -0.363% | -0.559% | -0.331% |

Positive reductions mean better. These are fitting diagnostics, not test
trajectory gains. In the second context, a lower average prediction error still
leads to more costly wrong decisions. The fixed internal follow-up screen fails
0/3, so no additional development or reserved-test evaluation is opened.

All six models complete 1,000 updates; resume, a full repeat and metric replay
agree exactly. Source-level analysis suggests the context change also removes
useful background information, but that is a hypothesis, not a proven cause.
The next model comparison will separate candidate identity from excluding peers
that fall back to the baseline. No deployment upgrade or safety exemption is
introduced, and the paper is not yet submission-ready.
[Results and loss curves](outputs/publication_readiness_2026_09/moving_proposal_context_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_proposal_context_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_proposal_context_v1/protocol.md).

### Last Completed Experiment

I completed six matched neural continuations to test whether emphasizing the
worst training source improves intervention decisions. Both arms use the same
trained starting models, inputs, query draws and 1,000-update budget. The only
change is a fixed 25% contribution from the worst training-source loss.

**Lower worst-source loss did not produce consistently better decisions.**

| Context | Worst-source loss reduction | Total decision-cost reduction | Harmful-admission-cost reduction |
|---|---:|---:|---:|
| 0 | +2.149% | -1.465% | -5.046% |
| 1 | +6.936% | -8.233% | +31.409% |
| 2 | +1.980% | +1.160% | -22.435% |

Positive reductions mean better. These are fitting diagnostics, not test
trajectory gains. The three worst sources improve, but the five other fitting
sources worsen. None passes the fixed internal follow-up screen, so I did not
run another development or reserved-test readout. Resume, a full refit and all
fitting metrics replay exactly; that verifies execution, not scientific success.

The original 2% risk and support requirements are unchanged. Empty support is
still not estimable, there is no identity-fallback exemption, and no model is
promoted. The next model question concerns representing the complete admission
decision more directly, rather than another source-weight or threshold sweep.
[Results and loss curves](outputs/publication_readiness_2026_09/moving_source_robust_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_source_robust_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_source_robust_v1/protocol.md).

### Latest Model-Input Diagnostic

I tested whether explicit relations between a candidate forecast and nearby
agents explain the remaining risk errors. Sixteen fixed linear probes leave out
one training source at a time. Adding these features worsens risk-prediction MSE
by 6.29%, 0.22% and 5.12% across the three contexts; decision cost improves in
only one. This does not support expanding that feature branch. It also does not
rule out nonlinear interaction models.

This motivated the source-risk training experiment above. These are exposed
training-side diagnostics, not a
new independent test or sixteen new neural networks. The original 2% rules,
support requirements and prior failures are unchanged; there is no identity-
fallback exemption or deployment upgrade.
[Diagnostic results](outputs/publication_readiness_2026_09/moving_candidate_neighbor_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_candidate_neighbor_v1/failure_analysis.md).

### Previous Neural Feature Experiment

I added explicit relations between a candidate forecast and the agent's past
motion: direction, speed change, acceleration alignment and departure from the
baseline. Six matched neural models test these32features with the same loss,
initialization, training budget and original safety rules.

**The features give a small development gain, but the policy is still unsafe.**

| Policy | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full original risk/support |
|---|---:|---:|---:|---:|
| Matched masked-feature control | 0.895% | 1.170% | 8.633% | 0/17 |
| Rollout-history relations | 0.981% | 1.303% | 8.660% | 0/17 |

The contrast is+0.0865percentage points, with missing-label completion bounds
[+0.0451,+0.1279]pp, not sampling confidence intervals. Only two of seven
sources improve; four worsen and one is unchanged. Most gain comes from one
source. Removing the decision head raises average gain but worsens maximum
easy degradation to25.726%, so that is not a deployment option.

All six fits, three original controls, a full refit and prediction/metric replay
are verified. These are already exposed obs8/pred12 image-coordinate development
results, not independent confirmation or new trajectory-dynamics training.
The original2% rules remain unchanged; no identity exemption or deployment
upgrade is introduced. The paper is not yet submission-ready.
[Results and loss curves](outputs/publication_readiness_2026_09/moving_rollout_history_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_rollout_history_v1/failure_analysis.md),
[protocol](outputs/publication_readiness_2026_09/moving_rollout_history_v1/protocol.md).

### Previous Paired Decision Result

I completed nine matched neural fits to test whether learning relative scores
for useful and harmful interventions improves the decision to replace a
baseline. The comparison keeps the same inputs, initialization, query draws,
training budget and original safety rules. No threshold was retuned.

**Better ranking did not produce a safer, more useful policy.**

| Frozen policy | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full original risk/support |
|---|---:|---:|---:|---:|
| Joint-decision control | 0.895% | 1.170% | 8.633% | 0/17 |
| Matched pointwise loss | 1.467% | 1.840% | 8.465% | 0/17 |
| Paired ranking loss | 0.836% | 1.129% | 8.337% | 0/17 |

The primary ranking-minus-pointwise contrast is **-0.6311 percentage points**.
Its missing-label completion bounds are [-0.8664, -0.3957] pp, not sampling
confidence intervals. Ranking improves fitting AUROC over the joint control
in all three contexts, but its decision cost is worse than the matched
pointwise arm in all three. The pointwise arm recovers more useful predictions
and increases harmful admissions; its larger mean gain does not make it safe.

These are partial-outcome results from seven already exposed development
sources, not independent confirmation or complete-roster ADE. All nine fits,
three predecessor controls, one complete refit and prediction/metric replay
are verified. Sixty-six focused tests and two completed-replay checks pass.
The read-only replay command also reproduces the real completed experiment
without changing any of its 36 then-existing public files. The resumed training took
28.6 minutes, excluding the interrupted process and downtime. No model is
promoted, no reserved scenes are opened, and no identity exemption is added.
The original 2% risk, easy and support requirements remain unchanged.
[Complete results](outputs/publication_readiness_2026_09/moving_paired_decision_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_paired_decision_v1/failure_analysis.md),
[reproducibility](outputs/publication_readiness_2026_09/moving_paired_decision_v1/validation_receipt.md).

### Checking The Failure Mechanism

I tested two specific explanations before extending training. Complete-fitting
gradients show risk/ranking conflict in two of three contexts, but a disposable
saved-AdamW projection improves MSE without raising decision cost in only one.
It does not support a full projection-training run. Separately, changing only
neighbor storage order changes many decisions in two contexts, but averaging
all eight fixed orders worsens fitting MSE and decision cost in all three.
This is not proof that neighbor order is a bug or that invariant models cannot
work; it rejects this particular shortcut. Both diagnostics replay exactly.

These are new own-fitting diagnostics of existing neural models, not additional
trained models, independent tests or safety passes. Original 2% rules, support
requirements and failed results remain unchanged. No identity exemption or
deployment change is introduced.
[Gradient evidence](outputs/publication_readiness_2026_09/moving_paired_gradient_v1/result_interpretation.md),
[neighbor-order evidence](outputs/publication_readiness_2026_09/moving_neighbor_order_v1/README.md).

### Previous Harm-Weight Allocation Result

I completed nine matched neural fits to test whether training should put more
weight on lower-magnitude harmful interventions. A uniform-weight control has
the same total positive coefficient mass, so the comparison tests allocation
rather than simply adding a larger penalty. The original 2% safety, reference
and support requirements remain unchanged.

**Rejecting more harmful predictions did not produce a better safe policy.**
The new loss reduces harmful admissions in every fitting partition, but also
rejects more useful predictions. Relative to the equal-mass control, total
decision cost increases in all three partitions.

| Frozen policy | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full original risk/support |
|---|---:|---:|---:|---:|
| Joint-decision control | 0.895% | 1.170% | 8.633% | 0/17 |
| Uniform harm weighting | 0.881% | 1.159% | 8.633% | 0/17 |
| Low-magnitude harm floor | 0.793% | 1.046% | 8.048% | 1/17 |

The primary utility contrast is **-0.0883 percentage points**, with
missing-label completion bounds [-0.1004, -0.0762] pp. These are not sampling
confidence intervals. Results cover seven already exposed development sources
and partial outcomes, not independent test confirmation or complete-roster ADE.
One passing recording view does not satisfy the full safety requirement.
Three selected-easy views remain undefined; no identity exemption is added.

All nine fits, exact predecessor controls, a complete treatment refit, metric
replay and own-fitting error attribution are complete. Fifty-one focused tests
pass; the full historical suite was not rerun. A receipt-export error was
recovered without changing the completed checkpoint. No model is promoted,
no reserved scenes are opened, and this experiment does not train new trajectory
dynamics. The next problem is distinguishing harmful from useful interventions,
not weakening the safety rule or increasing rejection indiscriminately.
[Complete results](outputs/publication_readiness_2026_09/moving_harm_floor_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_harm_floor_v1/failure_analysis.md),
[reproducibility](outputs/publication_readiness_2026_09/moving_harm_floor_v1/validation_receipt.md).

### Previous Joint-Decision Result

I completed six matched neural fits to test a concrete loss problem: a channel
with a zero target could block a useful intervention without receiving a
signed-cost gradient. Training the complete decision reduces this rejection
cost, but also admits more harmful interventions. The original 2% safety and
support rules were not changed.

**The model improves development utility slightly, but still fails safety.**

| Frozen policy | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full original risk/support |
|---|---:|---:|---:|---:|
| Previous per-channel model | 0.759% | 1.009% | 8.038% | 0/17 |
| Joint-decision model | 0.895% | 1.170% | 8.633% | 0/17 |

These are source-macro, partial-outcome results on seven already exposed
development sources, not independent test confirmation. The +0.136 percentage
point gain does not offset the safety failure. All predictions and metrics
replay exactly; that establishes reproducibility, not deployability. No model
is promoted, and no reserved test scenes were opened.

The audit also corrects an earlier explanation: training already used the
router's single proposed action. All-action averaging and opposing target
signs do not explain this failure. The remaining problem is distinguishing
useful interventions from harmful ones, not simply encouraging more switches.
[Complete experiment](outputs/publication_readiness_2026_09/moving_joint_decision_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_joint_decision_v1/failure_analysis.md),
[corrected diagnosis](outputs/publication_readiness_2026_09/moving_decision_semantics_v1/README.md).

### Previous Per-Channel Composition Result

I trained six matched models to test whether learning the combined risk and
intervention decision improves on training the two heads independently.
The original 2% limits, conservative labels and support rules remain unchanged.

**The loss change repairs fitting, but safe generalization is still unresolved.**
All three fitting contexts pass the prespecified internal screen. Compared with
independent training, all-decision cost falls 30.94%, 14.68% and 9.89%. Yet the
gain on already exposed development scenes is only +0.013 percentage points.

| Policy | Known ADE gain over floor | Worst defined easy-net degradation | Original full risk/support |
|---|---:|---:|---:|
| Independent two-head | 0.745% | 8.116% | 0/17 |
| Composed two-head | 0.759% | 8.038% | 0/17 |

These are partial-outcome development results, not independent test confirmation.
The original constant control has greater utility but also fails safety. Neither
the average gain nor lower training cost justifies deployment. No model is promoted.

All 12,000 primary updates, a 2,000-update exact repeat, prediction replay and
error attribution are complete. The three fresh controls exactly reproduce the
previous models. Thirty-six focused tests pass; the full historical suite was
not rerun. This is neural intervention-risk research, not new trajectory dynamics.
[Complete results](outputs/publication_readiness_2026_09/moving_composed_risk_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_composed_risk_v1/result_interpretation.md),
[run and recovery guide](outputs/publication_readiness_2026_09/moving_composed_risk_v1/operation_zh.md).

### Previous Separate-Representation Result

I trained nine models to test separate representations for risk magnitude and
intervention decisions, with narrow and near-capacity-matched shared controls.
The original 2% safety and support rules are unchanged.

**The change helps fitting in two contexts, but does not solve the problem.**
The separated model reduces MSE by 20.42%, 28.68% and 10.07%. Its all-decision
cost rises 4.13% in the first context and falls 30.71% and 20.18% in the other two. The failed
context admits fewer harmful predictions but rejects too many useful ones.
Separation also does not consistently outperform the wider shared control.

All 18,000 primary updates, a 2,000-update exact repeat and metric/error replay
are complete. Twenty-nine focused tests pass; the full historical suite was
not rerun. The prespecified fitting screen fails, so no additional held readout
is run and no model is promoted. This is risk-head research, not new trajectory
dynamics or independent confirmation. These findings motivated the composed
intervention loss experiment above, preserving the original rules and labels.
[Complete results](outputs/publication_readiness_2026_09/moving_decoupled_risk_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_decoupled_risk_v1/result_interpretation.md),
[run and recovery guide](outputs/publication_readiness_2026_09/moving_decoupled_risk_v1/operation_zh.md).

### Previous Complete-Fitting Result

I tested complete-fitting optimization of the neural risk model without changing
the original 2% safety or support rules. Three real fits, exact resume checks,
a full refit and error-attribution replay are complete.

**Better loss did not produce consistently better decisions.** Mean fitting
loss falls 31.98%, 37.90% and 41.13%, but every context worsens at least one
decision-cost channel. One model rejects more useful interventions; others
admit more costly harmful ones. These are training diagnostics, not test gains.
The prospective fitting screen fails, so no new held predictions are evaluated.
There is no deployment upgrade, changed threshold or identity certification.

Twenty-seven focused checks pass; the full historical suite was not rerun.
This is native obs8/pred12 SOURCE-development evidence, not independent testing
or a new trajectory-dynamics result.

This motivated the separate-representation experiment reported above.
The earlier complete-fitting result remains negative.
[Complete result](outputs/publication_readiness_2026_09/moving_full_objective_risk_v1/README.md),
[error analysis](outputs/publication_readiness_2026_09/moving_full_objective_risk_v1/result_interpretation.md),
[run and recovery guide](outputs/publication_readiness_2026_09/moving_full_objective_risk_v1/operation_zh.md).

### Previous Training-Schedule Result

I kept the original 2% safety and support rules and tested a training change.
Six matched neural risk-head fits compare a fixed learning rate with gradual
reduction, starting from identical weights and using the same inputs, labels,
loss and update budget. Full training, exact refit and prediction replay are
complete. A preceding fitting diagnostic did not support gradient conflict as
the local explanation for failure.

| Training schedule | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full risk/support |
|---|---:|---:|---:|---:|
| Constant | 3.60% | 4.63% | 38.90% | 0/17 |
| Gradually reduced | 3.75% | 4.75% | 44.65% | 0/17 |

**This is not a safe improvement.** The small average gain comes with more
easy-risk violations, and the original safety gate still fails. Full-fitting
risk errors also increase, so the experiment does not establish that the
optimization problem has been solved. I am retaining the negative result and
not deploying this model.

These are repeated development observations on seven SOURCE units and one
producer seed, not independent tests or new trajectory dynamics. No reserved
data, thresholds, support rules or identity-fallback certificates changed.
Thirty-four focused tests pass; the full historical suite was not rerun.
[Complete results](outputs/publication_readiness_2026_09/moving_annealed_risk_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_annealed_risk_v1/result_interpretation.md),
[reproducibility](outputs/publication_readiness_2026_09/moving_annealed_risk_v1/reproducibility.md).

### Completed Scale-Input Ablation

I tested whether explicit pixel scale misleads the neural intervention-risk
model. Six real retrained models, a full refit and exact prediction replay
are complete. I removed only the risk head's log-scale input and retained
the original 2% safety and support requirements.

| MSE risk head | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full risk/support |
|---|---:|---:|---:|---:|
| Original input | 3.60% | 4.63% | 38.90% | 0/17 |
| Explicit scale removed | 4.15% | 5.45% | 47.59% | 0/17 |

**Average utility improves, but the safety repair fails.** The matched gain
is +0.5525 percentage points. Some scenes benefit, another loses useful
interventions, and the worst easy-case degradation increases. The secondary
signed-cost comparison shows the same aggregate utility/safety tradeoff.
Neither version is deployed, and neither passes its expansion screen.

These are repeated development results on seven SOURCE units and one producer
seed, not independent confirmation. Missing-reference and empty-control cases
remain NA. Removing one input is not full-policy coordinate invariance, and
this experiment does not train new trajectory dynamics. Evaluation rules and
reserved data are unchanged.
Thirty-one focused tests pass; the full historical suite was not rerun.
[Complete comparison](outputs/publication_readiness_2026_09/moving_scale_ablation_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_scale_ablation_v1/result_interpretation.md),
[running and recovery](outputs/publication_readiness_2026_09/moving_scale_ablation_v1/operation_zh.md).

### Completed Signed-Cost Comparison

I tested a decision-aligned risk objective that charges mistakes according to
their realized cost. Six real neural fits, a complete treatment refit and exact
prediction replays are complete. The original 2% and support rules are unchanged.

| Risk objective | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full risk/support |
|---|---:|---:|---:|---:|
| Ordinary MSE | 3.60% | 4.63% | 38.90% | 0/17 |
| Signed-cost logistic | 2.69% | 3.56% | 26.23% | 1/17 |

**Better fitting does not establish safe transfer.** Training easy-channel
decision cost falls 81-85%, but held-source utility declines. Three sources
improve and four worsen; useful intervention nearly disappears in some scenes.
The one passing view contains only 13 selected rows. This is not an overall
safety pass or a reason to deploy the new model.

These are repeated development results on seven SOURCE units and one producer
seed, not independent confirmation or new world dynamics. Missing supports and
empty utility controls remain NA. No reserved data were opened or evaluation
rules revised. Twenty-six focused tests pass; deployment remains unchanged.
[Complete results](outputs/publication_readiness_2026_09/moving_signed_decision_v1/README.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_signed_decision_v1/result_interpretation.md),
[run and recovery guide](outputs/publication_readiness_2026_09/moving_signed_decision_v1/operation_zh.md).

### Completed Source-Held Error Test

I tested whether errors discovered on an unseen fitting source teach a better
intervention-risk model than errors discovered on an already seen source.
Eight discovery networks were trained from scratch, followed by six matched
risk-head continuations. All 14 models completed training, and full refits and
prediction replays reproduce exactly. The original 2% rules remain unchanged.

| Error discovery | Known ADE gain over floor | Hard gain | Worst defined easy-net degradation | Full risk/support |
|---|---:|---:|---:|---:|
| In-source control | 3.20% | 4.04% | 25.00% | 0/17 |
| Source-held errors | 2.95% | 4.06% | 64.13% | 0/17 |

**This repair does not establish safe utility.** The known paired contrast is
-0.2505 percentage points. Six sources improve, but a 5.03-point loss in the
seventh outweighs those gains. The method rejects many beneficial interventions
in one source while admitting beneficial but easy-harming interventions in
another. Even on the full fitting data, false-safe easy signs increase relative
to the control despite lower loss on the emphasized errors.

This is a development result on seven exposed SOURCE units and one producer
seed, not an independent test or a new world-dynamics contribution. No reserved
data were opened. Missing-support cases remain NA; no identity certificate,
deployment change or submission-readiness upgrade. Twenty-one focused tests
pass. I retain the negative result and the pre-training cache-interface repair.
[Full results](outputs/publication_readiness_2026_09/moving_source_error_priority_v1/README.md),
[mechanism and limitations](outputs/publication_readiness_2026_09/moving_source_error_priority_v1/result_interpretation.md),
[original-rule gates](outputs/publication_readiness_2026_09/moving_source_error_priority_v1/gates.md).

### Completed Fixed-Error Priority Test

I tested whether teaching the risk model to correct its earlier mistakes
improves safe intervention. Both models continue from identical weights;
one uses ordinary MSE, while the other gives its fixed, observed training
errors ten times the weight. The original 2% safety and support rules stay
unchanged. Six real neural fits, a full refit and exact replay are complete.

| Risk model | Known ADE gain over floor | Hard gain | Intervention rate | Full risk/support views passed |
|---|---:|---:|---:|---:|
| Warm kinematic model | 3.94% | 4.86% | 17.32% | 0/17 |
| Ordinary continued training | 3.60% | 4.63% | 15.01% | 0/17 |
| Fixed error-priority training | 3.03% | 3.85% | 13.59% | 0/17 |

**This objective repair fails.** The emphasized fitting-error loss falls
57.96-89.21% below the matched control, but held-source utility decreases.
The model learns many old mistakes, introduces new ones, and rejects useful
interventions. Both arms retain 14 all-risk and 13 easy-risk violations,
with two undefined selected-easy views. Worst defined easy-net degradation
is 22.67% for priority training, still far above the original requirement.

This is a controlled negative result on seven exposed SOURCE units and one
producer seed, not independent confirmation. Each fitting context has only
two or three source groups; many overlapping windows cannot substitute for
scene diversity. Neither more identical training nor this fixed error
weighting establishes safe utility. I have not changed the evaluation rules,
certified missing support, upgraded deployment or claimed a new dynamics method.
[Complete results](outputs/publication_readiness_2026_09/moving_admission_risk_v1/README.md)
and [what the experiment identifies](outputs/publication_readiness_2026_09/moving_admission_risk_v1/result_interpretation.md).

### Previous Kinematic Representation Test

I tested whether explicit causal motion features improve the neural model's
intervention-risk estimates. A 64-field basis describes past velocity changes,
turns, neighbor approach and baseline rollouts. I compared it with an otherwise
identical continued-training control in three matched contexts. Both arms
start from the same weights and use the same loss, sampling and update budget.
The original 2% risk and support requirements are unchanged.

Six real neural fits completed 12,000 primary updates. The new features reduce
full fitting loss by 30.13% to 72.99%, but do not improve overall held-source
utility:

| Risk model | Known ADE gain over floor | Hard gain | Intervention rate | Full risk/support views passed |
|---|---:|---:|---:|---:|
| Previous affine readout | 4.32% | 5.35% | 22.01% | 0/17 |
| Equal-budget continued-training control | 3.96% | 5.13% | 18.51% | 0/17 |
| Causal kinematic representation | 3.94% | 4.86% | 17.32% | 0/17 |

**This model repair does not pass.** Only one of three fixed contexts meets
the prospective expansion condition. Known all-risk violations increase from
13 to 14; easy violations fall from 16 to 14, but two selected-easy views also
become undefined. Worst defined recording easy-net degradation is 37.35%, not
a safety pass. I retain all contexts rather than promote the favorable one.

The fitting-policy comparison also worsens despite the lower loss. On the
actions it admits, the model underestimates easy harm even in fitting data;
unseen-scene shift alone cannot explain the failure. More training without the
new features also loses held utility against the warm model. These findings
point to a mismatch between average risk regression and conditional admission
decisions, not evidence that causal motion is generally useless.

Checkpoint resume, a full independent refit and all inference/decision/metric
replays are exact; 17 focused tests pass. This is new risk-encoder training,
not new trajectory dynamics. The comparison covers seven already exposed
SOURCE units and one producer seed. It is development evidence, not an
independent test. Missing-outcome completion bounds are not confidence
intervals. Deployment is unchanged; no identity-fallback certificate is used.
[Results, losses and failure analysis](outputs/publication_readiness_2026_09/moving_kinematic_risk_v1/README.md).

### Previous Nonlinear Readout Test

I tested whether a nonlinear risk model can recover useful information from
the frozen neural representation. In three matched contexts, I fitted one
readout using the two existing risk scores and another using those scores plus
64 hidden features. Inputs, source splits, proposals, supervision and the
original 2% risk requirements stay fixed.

The hidden-feature readouts reduce fitting loss by 40.87% to 64.59%, but that
advantage does not transfer to the seven exposed held SOURCE units:

| Risk readout | Known ADE gain over floor | Intervention rate | Full risk/support views passed |
|---|---:|---:|---:|
| Previous affine readout | 4.32% | 22.01% | 0/17 |
| Nonlinear scores only | 1.58% | 8.06% | 0/17 |
| Nonlinear scores + hidden features | 1.51% | 8.46% | 0/17 |

**This repair fails.** Both new readouts discard many useful interventions
while still admitting actions that exceed the harm budget. Adding hidden
features is slightly worse than the matched score-only control on held-source
known utility, despite helping on the fitting data. Known all/easy risk
violations remain 13/15, and two selected-easy views become undefined. Fewer
interventions and lower average prediction loss are not safety certificates.

Six real readout fits, checkpoint resume, a full independent refit and exact
inference/readout replay are verified. These are standard boosting readouts,
not new encoder training or a new dynamics method. The comparisons use one
producer seed on exposed development data; they are not independent-test or
multi-seed evidence. I have not changed the rules or upgraded deployment.
[Results and mechanism analysis](outputs/publication_readiness_2026_09/moving_nonlinear_readout_v1/README.md).

### Previous Affine And Direct Risk Tests

An exact affine-output refit previously raised known floor gain from 4.09%
to 4.32%, but left full risk/support at 0/17 and failed its expansion criterion.
That control helps distinguish final-layer optimization from the unresolved
problem of predicting risk on the subset selected for intervention.
[Affine control](outputs/publication_readiness_2026_09/moving_readout_refit_v1/README.md).

I tested a more direct way to learn intervention risk: predict how far an
action lies above or below the original harm budget. Six neural heads compare
ordinary mean regression with a fixed upper-quantile objective, with matched
initialization, sampling and 2,000 updates per head. This is a deliberately
limited model-repair experiment on seven already exposed SOURCE units, not an
independent benchmark or a full multi-seed result.

| Policy on the same pilot sources | Known ADE gain over floor | Intervention rate | Full risk/support views passed |
|---|---:|---:|---:|
| Previous proposal-conditioned critic | 5.33% | 31.79% | 1/17 |
| Direct mean-risk head | 4.09% | 21.31% | 0/17 |
| Direct upper-quantile head | 0.15% | 1.52% | 2/17 |

Neither repair is deployable. The upper-quantile head discards most useful
interventions yet still violates the original risk requirements. The problem
also appears on its fitting data, so unseen-scene shift is not the only
explanation. Marginal prediction quality does not ensure accurate risk on the
subset a model chooses to admit. I retain the failed and undefined cases,
including the quantile policy's two unavailable matched utility comparisons.

All six fits, the independent full refit, checkpoint resume and readout replay
are verified. The original 2% rules remain unchanged; there is no identity-
fallback certificate or deployment upgrade. The pilot does not meet its
prespecified condition for expansion.
[Results and failure analysis](outputs/publication_readiness_2026_09/moving_direct_tail_risk_pilot_v1/README.md).

### Previous Full Comparison

I completed a controlled test of proposal-conditioned risk learning: the risk
model learns only the actions proposed by a router fitted on different source
scenes. I trained 108 neural critics for 216,000 updates, keeping the forecaster,
initialization, sampling, source splits and original 2% requirements fixed.
Training took about 128 minutes locally, with exact checkpoint and inference
reproduction.

The change did not improve the policy. Compared with the all-action critic,
the known-label utility difference is -0.29 percentage points (95% source CI:
-0.86 to +0.28). Only 28/696 repeated recording views meet the full original
risk/support requirements, down from 38/696. Easy-risk violations remain high,
and eight required source views still lack an estimable matched comparison.

The failure is informative but not a success claim: the critic overestimates
harm across the proposal pool, then underestimates it on the actions it admits.
Restricting its training actions did not resolve that problem. **The original
rules and previous failures are unchanged, and I am not upgrading deployment.**
These are exposed MovingDroneCrowd++ development results, not independent
confirmation or a new safe world-model contribution.
[Full results, loss curves and failure analysis](outputs/publication_readiness_2026_09/moving_proposal_risk_v1/README.md).

## Previous Controlled Tests

An earlier comparison tests a published cost-sensitive routing objective on the
same frozen MovingDroneCrowd++ forecast bank. I trained 54 source-isolated neural
routers, with 108,000 updates and three producer seeds. The original point
controls and all new inference blocks reproduce exactly.

Direct routing improves available-label ADE over the physical floor by 12.29%
(95% source-bootstrap CI: 6.94 to 17.88). Against the previous point selector,
the paired known-label gain is 5.25 percentage points [1.31, 9.55]. Those are
exposed development results, not independent confirmation. The direct
missing-outcome lower comparison against the point selector still crosses zero.

The safety result is less encouraging. With the same risk veto, the paired
advantage is only 0.82 points [-1.05, 2.93], and just 38/696 repeated recording
views meet the original complete risk/support contract. There are 483 known
easy-risk violations. Eight required source views have empty matched controls,
so that complete comparison remains unavailable. **I have not changed the 2%
rule, certified empty cases as safe, or upgraded deployment.**

This control shows that better mean routing utility does not automatically give
reliable admission decisions. It adapts an existing public method; it is not a
new M3W loss or a new trajectory-world-model contribution by itself.
[Full results, loss curves and failure analysis](outputs/publication_readiness_2026_09/moving_public_deferral_v1/README.md).

### Joint Harm Supervision

The preceding controlled experiment tests whether explicitly learning the chance
and size of harm makes neural interventions more reliable. I trained 108
matched cost heads on MovingDroneCrowd++ SOURCE, with the same initialization,
sampling, fitting sources and update budget in both arms. Event supervision
improves harm-estimation errors over the matched product-moment model, but
does not solve the decision problem. Its occurrence probabilities are also
worse than fitting-source frequency priors on the held development sources.

At the same intervention count and amplitude distribution within each query,
the new veto loses 1.03 percentage points of known-label utility against its
matched control (95% CI: -1.89 to -0.37). The complete nonempty primary remains
unavailable because 24 required source views have empty controls. Only 47/696
repeated recording views meet the original risk/support contract. I retain
those failures rather than changing the 2% limit or certifying empty cases as
safe. Deployment is unchanged.
[Complete experiment, loss curves and failure analysis](outputs/publication_readiness_2026_09/moving_joint_harm_v1/README.md).

This follows a [negative source-separated critic experiment](outputs/publication_readiness_2026_09/moving_source_separated_critics_v1/README.md).
The remaining problem is not simply average error prediction: risk estimates
can be conservative over the candidate pool yet optimistic on the actions
the model admits. These are exposed development results, not independent
confirmation or a new trajectory-world-model success.

I completed nine fixed-budget EqMotion forecasting fits on the EuropeanSquares
SOURCE data, using three real training seeds and the same observations, labels
and training exposure as the registered PCT controls. This is a fixed-head K=1
adaptation of an existing public method, not a new architecture or reproduction
of published best-of-20 scores.

On observed labels, EqMotion improves available-label ADE relative to CV by
17.21% on the original SOURCE population. The stronger PCT controls matter:

|Comparator|SOURCE target windows|EqMotion minus control, known gain pp [95% CI]|
|---|---:|---:|
|Original PCT, registered primary|318,969|+14.52 [+9.55, +22.19]|
|Dimensionless agent-track PCT|318,969|+8.75 [+6.09, +11.93]|
|Current history-detector PCT|149,510|+4.68 [+1.39, +7.41]|

The last comparison uses the unchanged controller whitelist: EqMotion gains
10.81% versus CV there, compared with 6.14% for current PCT. The two populations
are not pooled. Differences are percentage points using a common known CV
denominator within locality, not percent error reduction relative to each PCT.
Intervals use 3,000 locality resamples after producer-fold and seed averaging.

All prescribed forecasts and 216 primary/secondary metric groups replay.
However, these are exposed SOURCE crossfit results, not independent final tests.
All missing-outcome lower intervals cross zero, and locality 082 loses to the
current PCT on known labels. **I have not upgraded deployment or claimed a new
safe world model from this result.** It provides a stronger proposal for the
next admission experiment.

![Observed SOURCE gains and unresolved missing outcomes](outputs/publication_readiness_2026_09/european_eqmotion_v1/figures/strong_controls.svg)

[Complete forecasting comparison](outputs/publication_readiness_2026_09/european_eqmotion_v1/readout.md),
[stronger controls](outputs/publication_readiness_2026_09/european_eqmotion_v1/strong_controls/readout.md),
[model/data card](outputs/publication_readiness_2026_09/european_eqmotion_v1/model_data_card.md)
and [failure analysis](outputs/publication_readiness_2026_09/european_eqmotion_v1/failure_analysis.md).

### Moving-Camera Source Check

I have also completed the matched forecasting controls on 64 fit-admitted
MovingDroneCrowd++ recordings. Both model families use the same causal history,
neighbors and fitting budget: nine PCT and nine fixed-head EqMotion fits, with
three actual seeds and 72,000 primary updates. Every saved forecast replays
exactly, as do two complete first-model refits.

There are 195,286 past-eligible target windows, including partial and missing
future labels. Only 58 recordings in 23 source groups support the fixed 8/12
task; one declared group has no usable window. The original 24-group primary
is therefore not estimable. I report the supported-group results as secondary
development evidence, not an independent final test.

|Model|Available-label pixel ADE|Mean groupwise gain over causal CV|
|---|---:|---:|
|Causal CV|61.3680|0.00%|
|PCT|77.3815|-39.27%|
|Fixed-head EqMotion|53.4925|+15.74%|

EqMotion's known-label paired advantage over PCT is +55.02 pp
[+27.89, +87.39], but the missing-outcome lower contrast is +14.46 pp
[-8.19, +38.93]. That unresolved interval matters: this is a stronger proposal,
not a passed safety gate. Native annotation steps are not verified video time,
and the derived forecasting task is not the dataset author's published task.

[Full new-source comparison and loss curves](outputs/publication_readiness_2026_09/moving_source_forecast_supported_v1/readout.md)
and [relative-motion diagnostic](outputs/publication_readiness_2026_09/moving_relational_diagnostic_v1/readout.md).
I also tested past-only shared-motion extrapolators, selecting their coefficients
on producer-fitting sources only. They do not beat CV on held sources: the
selected control's mean groupwise gain is -2.58%. EqMotion retains an observed
advantage, but its paired missing-outcome lower interval still crosses zero.
The [expanded baseline comparison](outputs/publication_readiness_2026_09/moving_social_causal_baselines_v1/readout.md)
keeps that limitation explicit rather than presenting a larger candidate list
as proof of a stronger baseline.
The subsequent matched cost study is also complete: 36 real Torch heads,
72,000 updates and exact full refit/replay checks. Adding 48 observed-motion
fields does not establish a gain: the known paired effect is -0.41 pp
[-2.66, +2.00], and the completion-lower effect is -5.33 pp [-8.95, -2.27].
Only 9/348 repeated recording views satisfy the shared-motion joint policy's
full original risk contract. It is not deployed.

The [failure diagnosis](outputs/publication_readiness_2026_09/moving_shared_motion_cost_v1/failure_analysis.md)
finds severe underprediction specifically on policy-selected samples, already
within critic-fitting sources. Global cost fitting can look reasonable while
the apparently safest choices are wrong. This makes selection-conditional risk
learning the next problem to solve, rather than simply adding more motion cues.

An equal-budget follow-up tests stable logit supervision from the same 18
frozen cost heads. It improves fitting on the old selected cases, but the new
policy still underestimates harm on its own choices. Its switch rate falls
from about 70% to 12%, while known utility falls by 9.31 pp against ordinary
continuation. The paired missing-outcome lower effect is -16.86 pp
[-27.30,-7.69], and the complete risk contract still fails. I do not deploy it.
The [matched continuation report](outputs/publication_readiness_2026_09/moving_decoder_link_repair_v1/readout.md)
and [failure analysis](outputs/publication_readiness_2026_09/moving_decoder_link_repair_v1/failure_analysis.md)
show why improved cost fitting is not the same as safe intervention.

The next check exposes a more basic limit of the fixed binary proposal.
Even a label-aware choice can satisfy the original complete recording gate
in only 299/348 repeated views. In 15 views there is no observed easy event;
in another 34 every legal easy neural prediction exceeds the 2% harm ratio.
These involve 14 of 58 actual-query recordings, not 49 independent sites.
I keep all of them in the result rather than filtering to the feasible cases.
The [exact feasibility check](outputs/publication_readiness_2026_09/moving_record_feasibility_v1/readout.md)
shifts the next work toward candidate quality and event support, not another
blind selector-loss sweep. It is a diagnosis, not learned model performance.
Fixed past-only intermediate forecasts then expand that label-aware opportunity
to 330/348 views, covering 31 of the 34 candidate-limited cases. However, every
uniform amplitude still fails the complete contract. The
[amplitude controls](outputs/publication_readiness_2026_09/moving_amplitude_opportunity_v1/readout.md)
motivate learning the size of an intervention as well as its occurrence;
they are not a deployed selector or a safety guarantee.

I tested that idea with 36 fresh amplitude-conditioned cost heads and 72,000
updates. Both controls see the same paths and have the same capacity; one
learns/selects only the full neural action, while the other learns/selects
intermediate amplitudes. All new outputs and policy scores replay exactly.

The matched joint-policy known effect is only +0.46 pp [-0.48,+1.49]; its
paired missing-outcome lower effect is -1.16 pp [-2.91,+0.22]. With the same
amplitude-head weights, the richer action menu has a small observed benefit
of +0.59 pp [+0.30,+0.92], but its missing-outcome lower interval crosses zero.
Neither finding supports a new safe deployment: only 10/348 repeated recording
views meet the full original risk contract. The richer menu increases the
source-average switch rate from 63.24% to 70.93%, rather than simply shrinking
every intervention. I retain the failed primary and do not deploy this head.

[Complete amplitude-learning results and loss curves](outputs/publication_readiness_2026_09/moving_amplitude_cost_v1/readout.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_amplitude_cost_v1/failure_analysis.md)
and [reproduction guide](outputs/publication_readiness_2026_09/moving_amplitude_cost_v1/operation_zh.md).
The next question is why selected harm remains underpredicted on fitting
sources, not whether another held-threshold sweep can make this result pass.

The [frozen fitting-curve check](outputs/publication_readiness_2026_09/moving_amplitude_shape_v1/readout.md)
now finds that roughly 70% of the supervised amplitude head's repeated fitting
rows violate positive-harm convexity,although the true-label curves do not.
This is not a unique cause of failure. I tested a same-capacity constrained
decoder with 18 fresh heads and 36,000 updates, retaining the verified old
amplitude heads as matched controls. Curve checks pass, but known utility is
essentially unchanged: -0.02 pp [-0.53,+0.51]. The paired missing-outcome lower
effect is -1.83 pp [-3.62,-0.42], and complete risk support stays at 10/348
repeated recording views. I do not deploy it or claim new dynamics lift.

[Complete constrained-head results and loss curves](outputs/publication_readiness_2026_09/moving_convex_amplitude_v1/readout.md)
and [failure analysis](outputs/publication_readiness_2026_09/moving_convex_amplitude_v1/failure_analysis.md).
This separates a correct structural prior from reliable intervention costs:
enforcing the right curve shape is not enough to learn the right magnitudes.

I followed this with 54 small tree refiners that keep the forecasts and
reference costs frozen. The experiment compares global, fixed-region and
refreshed-region learning of intervention harm. A same-selection comparison
confirms that the new estimator corrects much of the extreme fitting error;
the change is not just avoiding difficult samples.

That does not yet make the controller reliable. Refreshed weighting gains
0.070 percentage points over fixed weighting on known labels, but the paired
completion-lower interval still crosses zero. Worst signed easy-net degradation
is 1.37%, while positive-harm and support requirements still fail. Only 47 of
348 repeated recording views meet the complete contract. All 72 held cost
outputs and policy metrics replay exactly. I retain this as a diagnostic
result, not a new neural world model or deployment upgrade.

[Complete experiment](outputs/publication_readiness_2026_09/moving_harm_refinement_v1/readout.md),
[failure analysis](outputs/publication_readiness_2026_09/moving_harm_refinement_v1/failure_analysis.md)
and [common-selection checks](outputs/publication_readiness_2026_09/moving_harm_refinement_v1/fitting_check/readout.md).

I tested whether past/current crowd information helps this cost
estimator. The new comparison retrains equal-width local-zero and shared-context
controls, leaving forecasts, targets, data roles and the 2% requirements fixed.
All 36 refiners and both complete refits are finished. The complete held
readout does not establish a feature gain: shared minus local is -0.042 pp
on known labels, and the direct completion-lower effect is -0.054 pp with a
95% interval crossing zero. Shared context improves worst signed easy-net
degradation to 0.90%, but the complete risk contract still fails. All 54 cost
outputs and 108 decision metric blocks replay exactly. This remains development
evidence, not a deployment upgrade.
[Protocol](outputs/publication_readiness_2026_09/moving_shared_harm_refinement_v1/protocol.md).
[Complete results](outputs/publication_readiness_2026_09/moving_shared_harm_refinement_v1/readout.md)
and [failure analysis](outputs/publication_readiness_2026_09/moving_shared_harm_refinement_v1/failure_analysis.md).

The follow-up fixed-decision decomposition identifies reference support as a
separate problem: 47 active views lack observed selected easy reference even
though every selected row has some future label. I tested a two-part reference
head against direct regression with source-excluded internal validation. The
direct head is better; the event-probability head does not establish a Brier
improvement. C-selected reference prediction beats a constant prior, but loses
to the existing learned neural reference on both known-label and conservative
bounded targets. The original neural reference remains the stronger component;
better-than-prior loss is not enough. Forecasts and policies stay fixed; this
is not a deployment upgrade.
[Risk-factor diagnosis](outputs/publication_readiness_2026_09/moving_record_risk_factors_v1/readout.md)
and [complete learned-reference comparison](outputs/publication_readiness_2026_09/moving_hurdle_reference_v1/strong_control/readout.md).

A fixed-selection follow-up finds that lower row-level reference loss can
coexist with worse recording-level mass estimates. I tested a targeted
change to the existing neural cost objective: learn aggregate excess on its
fitting teacher's selected sets, while leaving the network, observations,
forecasts and risk requirements unchanged. This is an experiment, not a new
deployment result. The fitting-only diagnosis then found that the model was
more optimistic on its own selections than on the teacher's selections. I
therefore trained one matched follow-up using the current detached selection
in that extra loss. Both banks are now frozen: 36 new neural cost heads across
three seeds, with the same forecasts, inputs and training budget. Their
held-source comparison was fixed in advance. The frozen-teacher loss improves known
utility against the conservative tree, but loses to the matched neural head,
does not establish the completion-lower gain and badly fails recording-level
easy preservation. The current-selection version is also complete. It recovers
known utility over that failed loss, but still loses to the original neural
head and does not repair risk. Its own fitting-region harm estimates are even
more optimistic. I do not deploy either version or interpret successful
training and replay as a successful method. The next test should separate
proposal selection from risk evaluation on genuinely source-excluded fitting
data, rather than assume another loss sweep will solve this.
Before retraining, I also tested a second frozen head as a risk veto and
matched its intervention count in every query. That cheaper alternative loses
utility even on C: it removes some harm but rejects more useful predictions.
Both heads share C training, so this is not a test of an independently fitted
critic. I do not scale or deploy this particular veto based on a lower harm
numerator.
[Same-count veto result](outputs/publication_readiness_2026_09/moving_cross_critic_veto_v1/readout.md).
[Complete two-study analysis](outputs/publication_readiness_2026_09/evidence_manuscript_v9/selected_record_objective_addendum.md)
and [current-selection results](outputs/publication_readiness_2026_09/moving_adaptive_record_cost_v1/readout.md).

A separate verified support check also limits what training can establish:
15 of the 348 recording views have zero full observed easy-reference mass.
No selected subset can create positive reference mass in those fixed views.
I retain them as unsupported rather than claiming that a smaller learning
loss makes the original complete-roster risk requirement pass.
[Support diagnosis](outputs/publication_readiness_2026_09/moving_support_ceiling_v1/readout.md).
[Selected-set evidence](outputs/publication_readiness_2026_09/moving_selected_reference_v1/readout.md)
and [matched neural objective](outputs/publication_readiness_2026_09/moving_record_excess_cost_v1/protocol.md).

## What Has Not Worked

Better forecasting, better cost fitting and safe admission are different tests.
The existing controller improves average observed error but does not satisfy
the complete recording-level risk requirements. Recent controlled experiments narrow
the explanation without turning it into a success:

- Eighteen matched cost forests fail the registered paired lower contrast and
  complete risk contract. Reference sharing and projection slightly help on
  SOURCE; the fitting gap is already present before the decoder.
- Doubling the neural critic width reduces its full SOURCE fitting objective
  by only 0.43%. Its registered lower contrast is -0.636 pp, and only 9/252
  required recording/pipeline views pass. It is not deployed.
- The completed 102-recording calibration experiment finds no common risk
  multiplier that satisfies the original complete contract. Empty selected
  reference support is not counted as a measured safe zero.
- Explicit observation-support cost models reduce some risk violations but
  lose 2.56 pp of observed utility against the frozen direct head. Their full
  risk check still fails; better label-availability prediction is not a policy win.
- A complete, capacity-matched interaction-feature test lowers the fitting
  objective by about 11.5%, but its held-scene primary remains inconclusive:
  -0.021 pp [-0.615, +0.648]. Only 110/444 repeated recording/pipeline views
  meet the complete risk contract. Better latent features are not yet a safe
  admission result.
- On the new moving-camera source, relative-motion prediction improves on
  observed labels, but its unknown-outcome lower contrast is -24.87 pp
  [-46.62, -5.83]. This does not establish a robust positive effect. The proxy
  is also blind to common translation and cannot replace absolute error or
  the full recording-level risk checks.
- A complete fitting-side audit finds that clipped inputs account for only
  0.95% of selected known easy-harm mass. Outliers exist, but this does not
  support another clipping-map training sweep as the main repair.

The current contract retains the 2% all/easy positive-harm and signed easy-net
requirements in every required recording, with finite support. A favorable
global mean or one easy-error statistic does not replace that contract.

The [results ledger](README_RESULTS.md) contains the full numerical record.
The [previous front page](README_RESEARCH_HISTORY_2026_11_03.md) is archived
verbatim; its pending states and proposed next steps are historical, not current.

## Protocol And Limits

The main observation protocol is **8 observed steps and 12 requested steps**.
The European-source experiment uses a stride of 12 raw frames. Historical
raw-frame `t+50` scores are a separate task and are not mixed into these results.

Future targets are supervision and evaluation only. Future endpoints, central
velocity, test-endpoint goals, future-validity masks and test normalization do
not enter inference. Producer, preprocessing and teacher exposure follow the
same data-role boundaries. Calibration private arrays, final confirmation and
DroneCrowd remain closed for the current experiments.

The project remains a **2.5D multi-agent trajectory/world-state research system**,
not true 3D or a foundation world model. The new forecasting control is 2D
image-plane geometry with time, not depth. Pixel/dataset-local coordinates are
not metric without verification; raw frames are not seconds. Detector/tracker,
inferred and self-audited labels are not human gold. Harm diagnostics are not
physical-safety guarantees. Stage5C execution and SMC remain disabled.

Earlier Stage35/37/43/44 results have recording, teacher-exposure or selection
limitations. They remain exploratory; new split names do not make them
independent evidence. The stronger SOURCE forecaster is a positive component
result, not yet a submission-ready M3W method.

## Running And Reading

Use native arm64 `.venv-pytorch` on Apple Silicon, not x86 Conda under Rosetta.
The completed European study uses four CPU compute threads, one inter-op thread and no
DataLoader workers, with atomic checkpoints, heartbeat and resume. It completes
36,000 primary updates, a 4,000-update full refit and a 200-update recovery pilot.
Training/refit takes 3.58 h; complete primary inference replay takes 2.10 h.
These are real Torch runs, not NumPy fallback. MPS success is not claimed.

Public scalar reports can be rebuilt without rereading raw label arrays:

```sh
.venv-pytorch/bin/python scripts/report_m3w_european_eqmotion.py
.venv-pytorch/bin/python scripts/report_m3w_european_eqmotion_controls.py
```

The full pipeline entry points are in the experiment protocol/readout. Completed
stage receipts are immutable: resume incomplete work and do not delete results
to manufacture a fresh run. Full execution still requires verified upstream
data and private checkpoints; anonymous cold-start reproduction remains a gap.
The earlier European scoped suite had 75 passing tests and one optional Metal-parity skip.
New-source checks and actual full replays are recorded with each experiment.
The unrelated legacy suite is not presented as a clean read-only smoke test.

|Record|Purpose|
|---|---|
|[Results ledger](README_RESULTS.md)|Detailed outcomes and negative results|
|[Research state](research_state.json)|Current status and next action|
|[Working paper](outputs/publication_readiness_2026_09/evidence_manuscript_v9/README.md)|Evidence draft, not submission-ready|
|[Data-role contract](outputs/publication_readiness_2026_09/experiment_contract/implementation_and_limits.md)|Role and exposure boundaries|
|[Local/CREATE guide](outputs/publication_readiness_2026_09/local_create_runbook_zh.md)|Runtime and recovery guidance|
|[Historical lineage audit](outputs/publication_readiness_2026_09/recording_lineage_audit.md)|Limits of earlier claims|

Raw datasets, feature/history/latent caches, large checkpoints, videos,
third-party images and local virtual environments stay out of Git.

## Next Question

I am checking MovingDroneCrowd++ as a possible new development source rather
than reusing protected confirmation recordings. Its annotation intake contains
120 clips and 245,018 complete native 8/12 windows, not that many independent
scenes. All 360 sparse source views are now available; 21 portrait views needed
a fixed axis correction before matching. All 20,580 source-overlap pairs are
now complete, with full saved-result verification and a repeatable read-only
check. The 69-minute matching job hit a summary-assembly error after saving
every pair; a registered repair recovered the work without changing matching.

The screen finds two strong and 29 ambiguous pairs within the new source,
but no supported cross-source pair. This is not proof of independent sites.
The 64 publisher-train candidates retain 136,295 complete native 8/12 windows;
they are not independent samples. At the source-screen checkpoint they were
not yet admitted to training.
No new forecasting score or deployment result follows from this audit.
The DUT bridge is now complete: all 28 author backgrounds (19.4 MB), 6,874
fixed pairs and exact source/receipt replays. No supported cross-source link
or reservation conflict was found, with crop/composite/matching limits retained.
A background composite is not a past video frame and will not become a causal
model input.

The separate [fit-only admission](outputs/publication_readiness_2026_09/moving_development_admission_v1/readout.md)
now opens those 64 clips for local development fitting. Their 24 conservative
groups have a fixed proposed threefold allocation of 45,606 / 45,359 / 45,330
complete windows. Other clips remain unadmitted for prediction in this track;
DUT and DroneCrowd retain their original roles. The next experiment must bind
the actual predictor/teacher/critic exclusions before fitting. Source access
is ready; a new neural benefit or safe policy has not yet been demonstrated.
The [first admitted-source forecast comparison](outputs/publication_readiness_2026_09/moving_source_forecast_v1/protocol.md)
is now registered: matched PCT and fixed-head EqMotion, three source folds and
three actual seeds, 18 complete 4,000-update fits. It includes partial and
wholly missing future labels rather than keeping only complete trajectories.
Its matrices will stay in RAM; a real-data optimizer/resume pilot precedes
the full run. No new training result is claimed at registration.
Preparation now exposes one empty source group: its three clips have only
10-17 native frames, too short for the fixed 8/12 request. All 195,286 data
rows are intact. A forward support repair keeps the original 24-group contrast
not estimable and labels the 23 supported-group contrast secondary; it does
not turn the missing group into zero error or shorten the task.
The real native-arm64 Torch pilot now passes for both families: 50+50-step
resume and direct100 produce identical model/optimizer/sampler states and
outputs. Peak RSS is 1.46 GB; measured speed projects about 3.1 hours for all
18 full fits. The complete matrix is running: all nine PCT producers have now
finished their full 4,000 updates, and EqMotion continues. Pilot prefixes are
reused only for their identical first primary jobs. Pilot loss is fitting evidence, not a held
prediction or risk result.
[Complete result](outputs/publication_readiness_2026_09/movingdronecrowd_background_v2/readout.md),
[inspection scope](outputs/publication_readiness_2026_09/movingdronecrowd_background_v2/pair_review.md),
[Chinese reproduction guide](outputs/publication_readiness_2026_09/movingdronecrowd_background_v2/tutorial_zh.md).

Can the stronger frozen proposal support useful, reliable admission rather
than another increase in critic width? The completed SOURCE experiments
narrow that question. The negative 082 slice and missing futures stay in the
analysis; none of these results changes deployment.

The [first amplitude experiment](outputs/publication_readiness_2026_09/european_eqmotion_amplitude_v1/readout.md)
is complete: all 1,440 policy evaluations replay exactly. The prespecified
quarter-amplitude EqMotion gains 6.28% over the fixed causal floor, but 0/444
recording views satisfy the full risk contract. Fixed support removes undefined
reference denominators; it does not remove actual easy harm or uncertainty from
missing futures. Even amplitude .001 fails six views of the same recording,
while gaining only 0.027%. No grid point is chosen for deployment.

The [causal label-support probe](outputs/publication_readiness_2026_09/european_source_label_support_v1/readout.md)
is also complete. On excluded SOURCE sites, past detector history raises AUROC
from 0.602 to 0.706; the paired feature effect is +0.104 [0.060, 0.162]. AUPRC
is 0.144 versus equal-locality unknown prevalence 0.042. Eighteen full forests,
real recovery/full refit, all probability arrays and 144 metric groups replay.
Training takes 267.97 s with 4.51 GB peak RSS. Future masks are targets only.
Calibration remains weak on the critical scene, and some scenes do not improve.
This is a release-label-process result, not physical visibility or neural dynamics.

The [matched bound-aware cost experiment](outputs/publication_readiness_2026_09/european_bound_aware_cost_v1/readout.md)
is complete: 36 real Torch heads, 116,160 primary updates, 4,000 full-refit and
1,200 recovery-pilot updates. Both fitting stages exclude each evaluation site;
all cost/amplitude arrays and 72 metric groups replay exactly. Training takes
378.71 s with 9.24 GB peak RSS. Explicit unknown-row targets lower their own
fitting objective from 4.884 to 1.489, but the direct paired primary is
**-1.70 pp [-5.71, +1.12]**, not a supported improvement. Known utility differs
by +0.38 pp [-1.48, +1.95], also inconclusive.

Full-risk support improves from 28/444 to 108/444 repeated recording views,
while known-all violations increase. Neither policy passes. In the worst
signed easy-net view, predicted easy risk is 0.060% but observed easy positive
harm is 57.76%. The [failure analysis](outputs/publication_readiness_2026_09/european_bound_aware_cost_v1/failure_analysis.md)
separates target-fitting progress from this selected-risk failure. I will next
decompose harm and reference errors on identical selected sets, on fitting and
excluded SOURCE sites, before choosing another cost-learning repair. No new
threshold will be selected from exposed calibration or final-confirmation data.

The [same-set cost diagnostic](outputs/publication_readiness_2026_09/european_bound_cost_decomposition_v1/readout.md)
is also complete, with full frozen-model replay. It locates failure before
new-scene transfer: on fitting sites, the bound-aware head's median selected
EH predicted/target mass ratio is only 0.0065. Every one of its 358 falsely-safe
fitting views underestimates EH; only 40 also inflate ER. Error is therefore
not just a denominator problem. Selection worsens the unattenuated EH mass
ratio in 423/444 fitting views. Fitting views with no wholly unknown selected
labels can also fail, so missing-label bounds alone do not explain it. This is a zero-update diagnosis, not a
new policy or another training success.

The [observation-support comparison](outputs/publication_readiness_2026_09/european_observation_cost_v1/readout.md)
is complete: 36 real Torch fits, 116,160 primary updates and exact full replay.
Training takes 404.41 s with 9.23 GB peak RSS. Its primary paired lower effect
is **-3.13 pp [-4.41, -1.99]**; observed utility also regresses,
**-2.56 pp [-3.74, -1.48]**. All twelve locality-averaged known effects are
negative. Risk support improves to 160/444, but worst easy-net degradation is
8.32%, so this model is not deployed.

The identified availability head learns more about released labels, but average
intervention amplitude falls from 0.239 to 0.091 and useful gains are lost.
The [failure analysis](outputs/publication_readiness_2026_09/european_observation_cost_v1/failure_analysis.md)
keeps that distinction explicit. The subsequent
[frozen interaction-feature experiment](outputs/publication_readiness_2026_09/european_interaction_feature_cost_v1/readout.md)
is also complete: 36 real cost-head fits, three actual seeds, exact replay of
all predictions and 72 metric groups. The two-hour feature extraction and
separate two-hour re-extraction preserve every original forecast. The added
feature bundle improves fitting, but its registered lower effect and known
effect both have intervals crossing zero. It is not a new safe policy.

The [new same-set diagnosis](outputs/publication_readiness_2026_09/european_feature_cost_diagnosis_v1/conclusions.md)
finds that this head underestimates selected easy harm already on fitting
scenes, not only on new localities. All 324 falsely-safe fitting views and 330
held views underestimate its numerator. I then tested whether more training
could fix this before changing the architecture again. The
[complete fourfold-budget experiment](outputs/publication_readiness_2026_09/european_feature_duration_probe_v1/conclusions.md)
finishes all 18 cost heads, 232,320 primary updates and a separate exact replay.
Average fitting loss improves 34.77%, but selected easy harm remains badly
underestimated. On the unchanged original decisions its prediction/target
ratio rises from0.0032 to0.0093; on the new model's own decisions it is0.0028.
The latter still violates easy risk in 307/444 repeated recording/pipeline
views. Longer training alone has not produced a safe learned policy.

The run recovered from a disk-reserve interruption without lowering the
budget or discarding completed heads. Both the original and completed
checkpoint sets are verified backups on CREATE; no remote training was
needed. The completed
[label-support decomposition](outputs/publication_readiness_2026_09/european_duration_support_diagnosis_v1/conclusions.md)
shows why missing labels are not the whole explanation. Unknown-row bounds
account for a median 64.26% of the evaluated easy-harm numerator, but 223 views also fail on
available labels. Even on fully labeled twelve-step paths, predicted easy
harm is only 0.83% of the target. Neither lower loss, fewer interventions nor
dropping uncertain labels establishes safe world-dynamics improvement.

I am now checking a different development-data route:
[MovingDroneCrowd++](outputs/publication_readiness_2026_09/movingdronecrowd_intake_v1/conclusions.md),
which provides image sequences and head-identity annotations. Its version is
pinned before acquisition. The completed annotation pass finds 120 clips,
638,718 labeled rows and 245,018 complete eight-observation/twelve-prediction
windows, using 7.18 MB of network payload rather than the full archive.
These are overlapping native-step windows, not independent samples or matched
physical time. The existing DroneCrowd collection remains
reserved for confirmation in its entirety. The new source is not yet admitted
for training or claimed to be independent of those reserved scenes. The next
step checks source overlap before assigning research roles.

The [registered image-overlap study](outputs/publication_readiness_2026_09/movingdronecrowd_background_v2/protocol.md)
covers every new clip and its relation to the protected DroneCrowd collection.
It keeps ambiguous matches as exclusion constraints. Passing a software check
or finding no image match will not be treated as proof of a new physical site.
Acquisition exposed 21 portrait views that need an explicit orientation fix
before matching; the original pass is retained and no training role is granted.

![Bound-aware cost fitting does not establish full policy safety](outputs/publication_readiness_2026_09/european_bound_aware_cost_v1/figures/cost_target_comparison.svg)

The larger goal remains useful neural dynamics backed by reproducible,
independent evidence. More stages or a lower training loss are not substitutes.
