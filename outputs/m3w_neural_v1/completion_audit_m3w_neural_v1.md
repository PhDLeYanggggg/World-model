# M3W-Neural v1 Completion Audit

- source: `fresh_run`
- completion_status: `not_complete`
- current_best_deployable: `M3W-Neural v1 composite-tail safe-switch bounded neural dynamics candidate under Stage37/teacher floor (bootstrap+multiseed+pure-UCY source-heldout and UCY-only policy-head supported; strict pure-UCY neural retrain still pending)`

## Requirement Matrix

| Requirement | Status | Evidence | Note |
| --- | --- | --- | --- |
| external split covers ETH/UCY/TrajNet or blockers | `complete` | outputs/stage41_external_split/report.json and Stage41 gates |  |
| no leakage: future endpoint label/eval only, no central velocity, no test endpoint goals | `complete` | outputs/stage41_breakthrough/stage41_endpoint_geometry_audit.json |  |
| neural model exceeds Stage37 on external all/t50/hard with easy <=2 | `complete` | outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json |  |
| at least two held-out external domains positive | `complete` | outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json |  |
| neural without external fallback not catastrophic | `complete` | fresh self-gated endpoint records no-external-fallback safe, but raw ungated endpoint remains unsafe in Stage41 reports | The self-gated neural output is safe; raw ungated endpoint dynamics remain unsafe and are not deployable. |
| all active agents future world-state, not only endpoint selector | `complete` | outputs/stage41_breakthrough/stage41_all_agent_eval.json, stage41_all_agent_risk_repair.json, stage41_all_agent_t50_specialist.json, stage41_all_agent_policy_composer.json, outputs/stage41_stratified_protocol/stage41_fixed_policy_confirmation.json, outputs/stage41_fresh_confirmation/stage41_fresh_all_agent_endpoint_specialist.json, outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json, outputs/stage41_fresh_confirmation/stage41_joint_latent_rollout.json, and outputs/stage41_fresh_confirmation/stage41_all_agent_composite_world_state.json | The fresh composite-tail all-agent audit applies the frozen composite-tail policy to full future waypoint rollouts for every active row in same-frame multi-agent groups. It passes ADE and FDE all/t50/t100/hard checks, preserves easy cases, improves multi-agent rows, and passes proximity/smoothness checks without future inputs. Ungated/no-fallback neural rollout remains unsafe and is not claimed. |
| full trajectory, interaction, occupancy, and physical-validity heads | `complete` | outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json | Trajectory ADE/t50/t100/hard improve with easy preserved; interaction and occupancy heads report AUROC/AUPRC. The separate goal/route/physical repair pass adds a non-degenerate physical-challenge label. |
| explicit goal/route head and non-degenerate physical-consistency target | `complete` | outputs/stage41_fresh_confirmation/stage41_goal_route_physical_repair.json | Route top1 beats majority and physical-challenge AUROC is high. Labels are still supervised future-waypoint targets, never inference inputs. |
| route/physical heads deployment contribution proven or disabled | `complete` | outputs/stage41_fresh_confirmation/stage41_route_physical_policy_integration.json, stage41_joint_route_conditioned_world_state.json, and stage41_route_physical_group_consistency.json | Auxiliary route/physical heads are predictive diagnostics. Post-hoc route/physical gating, joint route-conditioned training, and route/physical-augmented group consistency did not improve the deployable trajectory policy, so route/physical is disabled as a deployment contribution and kept diagnostic-only. |
| joint multi-agent consistency improves trajectory deployment policy | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_multiagent_consistency.json | Current-frame group consistency adds a tiny positive deployment-policy lift over the full-trajectory reference and gives UCY a small positive switch rate, but it is still post-hoc group calibration rather than a jointly consistent latent rollout. |
| neural gain/harm/switch distillation improves deployment without base-switch leakage | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation.json | The deployable distiller-only policy learns gain/harm/switch from train labels and uses past/static/full-trajectory prediction signals at inference. It improves ETH_UCY and TrajNet but still falls back on UCY. |
| no-base-switch distiller bootstrap and ablation evidence | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_evidence.json | Bootstrap lower bounds are positive for all/t50/hard; ablations show static causal features and full-trajectory prediction signals are the main positive contributors, while UCY remains fallback-only. |
| no-base-switch distiller multi-seed replication | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_multiseed.json | Three fresh seeds keep all/t50/t100/hard positive with easy preserved and two positive domains per seed; UCY remains fallback-only. |
| UCY fallback-only blocker diagnosed and repaired without test tuning | `complete` | outputs/stage41_fresh_confirmation/stage41_ucy_fallback_repair.json | UCY was missing from validation, so no UCY slice thresholds were selected. A train-only UCY calibration subset repairs UCY on test, but independent UCY validation is still needed before final deployment. |
| UCY repair internal fold/temporal validation | `complete` | outputs/stage41_fresh_confirmation/stage41_ucy_independent_validation.json | UCY repair validates on internal held-out row folds and temporal blocks. True source-level UCY validation remains unavailable because there is one UCY train source and no UCY validation source. |
| grouped all-agent rollout consistency under repaired policy | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_rollout_consistency.json | Audits same-frame multi-agent selected future waypoints for switch coherence, proximity risk, smoothness, and multi-agent improvement. This is grouped rollout evidence, not Stage5C latent generation or SMC. |
| joint latent all-agent rollout prototype trained and audited | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_latent_rollout.json | The group-token Transformer trains and auxiliary interaction/occupancy/future-close heads are useful, but deployment is disabled because raw neural rollout is FDE-negative and safe validation policy chooses fallback-only. |
| baseline-relative bounded residual rollout repair attempted | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_residual_rollout.json | Residual clipping reduces raw neural damage versus direct joint latent rollout, but the selected test policy is still all/t50/hard negative and not deployable. |
| domain/horizon residual policy repair attempted after global residual gate failed | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_residual_domain_policy.json | Validation-only domain/horizon slicing reduces switch rate and protects easy cases, but t50 remains zero and all/hard are not reliably positive, so it is not deployable. |
| teacher-guided neural proposal trained and evaluated without inference leakage | `complete` | outputs/stage41_fresh_confirmation/stage41_teacher_guided_proposal.json | A teacher-guided neural proposal learns from Stage37/group-consistency switch labels and neural proposal scores. Raw test gains are strong across all/t50/t100/hard, but the unguarded proposal exceeded the near-proximity safety delta and is not deployable by itself. |
| teacher-guided proposal safety repair passes deployment gates | `complete` | outputs/stage41_fresh_confirmation/stage41_teacher_guided_proposal_repair.json | Validation-selected proximity repair restores joint safety and still improves the current group-consistency multi-seed safety-buffer basis on all/t50/hard with easy=0. This is a strong single fresh run; multi-seed/CI is still required before freezing it as the final M3W-Neural v1 policy. |
| teacher-guided repair bootstrap CI and ablation evidence | `complete` | outputs/stage41_fresh_confirmation/stage41_teacher_guided_evidence.json | Frozen policy/guard evidence adds 2000-bootstrap confidence intervals and feature masking. CI lows are positive for all/t50/t100/hard and every external domain; ablations show group/neighbor consistency features are necessary. No-fallback neural remains unsafe for easy cases, so Stage37 safety fallback remains required. |
| teacher-guided repair multi-seed replication | `complete` | outputs/stage41_fresh_confirmation/stage41_teacher_guided_multiseed.json | Three fresh teacher-guided seeds each select policy and proximity guard on validation, then evaluate test once. All seeds are positive on all/t50/t100/hard, easy=0, joint collision delta below the safety ceiling, and all three external domains positive. |
| composite-tail bounded neural dynamics improves teacher repair | `complete` | outputs/stage41_fresh_confirmation/stage41_composite_tail_evidence.json | Composite-tail keeps the validation-repaired teacher switch set and adds a low-risk bounded neural tail; it improves the teacher repair on all/t50/t100/hard while preserving easy cases. |
| composite-tail bootstrap and multi-seed evidence | `complete` | outputs/stage41_fresh_confirmation/stage41_composite_tail_evidence.json and stage41_composite_tail_multiseed.json | Composite-tail has positive bootstrap lower bounds versus the floor and versus teacher repair, plus three seed-aware evaluations with positive all/t50/t100/hard deltas and easy=0. |
| pure UCY source-heldout frozen-policy validation | `complete` | outputs/stage41_external_split/stage41_pure_ucy_source_validation.json | Composite-tail policy is selected on non-UCY validation rows only and evaluated once on UCY zara01/zara02/zara03. This is not a pure UCY-only retrain/select/test protocol. |
| pure UCY train/val/test policy-head calibration | `complete` | outputs/stage41_external_split/stage41_pure_ucy_retrain_protocol.json | The ridge gain/harm policy head is trained only on UCY train rows, selected on UCY validation rows, and evaluated once on UCY test rows. It passes policy safety, but the underlying neural proposal and floor are still mixed-external trained, so strict pure-UCY neural retrain remains false. |
| domain-local neural endpoint retrain positive on at least two external domains | `complete` | outputs/stage41_domain_local/stage41_domain_local_neural_retrain.json | Fresh domain-local endpoint neural models are trained from causal seq2seq inputs with validation-selected gain/harm gates. ETH_UCY, TrajNet, and UCY_expanded are positive in endpoint-FDE, while the small default UCY split remains negative. This is endpoint-only evidence and does not replace the protected all-agent composite deployment path. |
| domain-local neural endpoint evidence passes same-frame all-agent safety proxy on at least two domains | `complete` | outputs/stage41_domain_local/stage41_domain_local_all_agent_world_state.json | The domain-local endpoint models are projected into endpoint-linear waypoint rollouts and audited for same-frame multi-agent ADE/FDE, proximity, and smoothness. ETH_UCY and UCY_expanded pass after validation-selected proximity guarding; TrajNet remains endpoint-positive but proximity-unsafe on test, and the small default UCY split remains negative. This strengthens safety evidence but is still not a learned full-waypoint rollout. |
| domain-local learned full-waypoint neural dynamics positive on at least two domains | `partial` | outputs/stage41_domain_local/stage41_domain_local_full_trajectory_world_state.json | Fresh learned full-waypoint domain-local dynamics were trained for ETH_UCY and TrajNet. The gate fails: ETH_UCY has all/hard ADE lift but no t50/t100 ADE lift and unsafe proximity delta; TrajNet has endpoint-FDE lift and t100 ADE lift but negative all/t50/hard ADE. UCY is blocked by no validation rows in the current all-agent split. This is negative evidence, not deployable. |
| neural group-consistency head improves joint-safe fixed proximity guard | `complete` | outputs/stage41_fresh_confirmation/stage41_group_consistency_distiller.json | Trains a neural safe-switch/gain/unsafe head from train labels and selects thresholds on validation. It improves the fixed proximity guard while preserving easy cases and joint proximity safety, but it is still a guarded selector/dynamics head rather than Stage5C latent generation. |
| group-consistency distiller bootstrap and ablation evidence | `complete` | outputs/stage41_fresh_confirmation/stage41_group_consistency_evidence.json | Bootstrap lower bounds are positive for all/t50/t100/hard. Ablations show the new group-consistency/proposal-score features are necessary, while some older feature blocks are not positive in this head. |
| group-consistency distiller multi-seed replication with joint-safety buffer | `complete` | outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed.json and outputs/stage41_fresh_confirmation/stage41_group_consistency_multiseed_repair.json | The first three-seed run had stable positive FDE gains but one seed exceeded the near-proximity delta threshold. A validation-selected safety-buffer repair passes all three seeds with positive all/t50/t100/hard, easy=0, and max collision delta below the joint-safety ceiling. |
| t100 diagnostic positive or blocker analysis | `complete` | outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json |  |
| JEPA contribution proven or disabled | `complete` | outputs/stage41_fresh_confirmation/stage41_jepa_deployment_decision.json | Current audited JEPA variants are non-collapse in several stages but do not produce deployable downstream lift. JEPA is disabled from the M3W-Neural v1 deployable path and kept diagnostic-only. |
| Stage5C disabled and SMC disabled | `complete` | outputs/m3w_neural_v1/package_manifest_m3w_neural_v1.json |  |
| no metric/seconds/foundation/true-3D overclaim | `complete` | outputs/m3w_neural_v1/report_m3w_neural_v1.md and data/model cards |  |

## All-Agent Risk Repair Result

- deployment_decision: `diagnostic_keep_m3w_neural_v1_endpoint_candidate`
- all improvement: `0.09976285280545372`
- t50 improvement: `-0.002800354643290648`
- t100 diagnostic improvement: `0.26476770940707695`
- hard/failure improvement: `0.10663942185551323`
- easy degradation: `0.0`

## All-Agent t50 Specialist Result

- deployment_decision: `diagnostic_keep_m3w_neural_v1_endpoint_candidate`
- all improvement: `0.023127391643180673`
- t50 improvement: `0.09375204966816386`
- t100 diagnostic improvement: `0.0`
- hard/failure improvement: `0.02472403070303797`
- easy degradation: `0.0`

## All-Agent Policy Composer Result

- deployment_decision: `diagnostic_keep_m3w_neural_v1_endpoint_candidate`
- best variant: `risk_all_t50_override`
- all improvement: `0.12271025390115187`
- t50 improvement: `0.0902220631231122`
- t100 diagnostic improvement: `0.26476770940707695`
- hard/failure improvement: `0.13117103605560632`
- easy degradation: `0.0`

## All-Agent Locked-v2 Fixed Confirmation

- deployment_decision: `candidate_needs_fresh_external_confirmation_before_deployment`
- stage37 margin pass: `True`
- stress pass: `True`
- fresh confirmation pass: `False`
- all improvement: `0.17133358603743754`
- t50 improvement: `0.2368852639341944`
- t100 diagnostic improvement: `0.19179621162648797`
- hard/failure improvement: `0.1782452385355816`
- easy degradation: `0.0`

## Fresh Source-Rotation All-Agent Endpoint Specialist

- deployment_decision: `fresh_all_agent_endpoint_candidate_needs_independent_acceptance`
- best name: `fresh_all_agent_endpoint_ensemble`
- all improvement: `0.26231894385271437`
- t50 improvement: `0.2754049021317291`
- t100 diagnostic improvement: `0.3011547102800356`
- hard/failure improvement: `0.2850203073377977`
- easy degradation: `0.0`
- positive external domains: `2`

## Full-Trajectory World-State Probe

- deployment_decision: `candidate_full_trajectory_world_state_probe`
- best name: `full_trajectory_ensemble`
- trajectory ADE all improvement: `0.18577852429834418`
- trajectory ADE t50 improvement: `0.14803699577731477`
- trajectory ADE t100 diagnostic improvement: `0.22857426649949408`
- trajectory ADE hard/failure improvement: `0.19518047277951456`
- easy degradation: `0.0`
- positive external domains: `2`
- interaction AUROC: `0.9614642176190807`
- occupancy AUROC: `0.9486653948303418`

## Goal/Route And Physical-Consistency Repair

- pass gate: `True`
- best name: `goal_route_physical_hard`
- route top1: `0.7590404840801037`
- route top3: `0.9820631032992364`
- route majority top1: `0.5532884310618067`
- route lift over majority: `0.20575205301829702`
- physical challenge AUROC: `0.9523668831032517`
- physical challenge AUPRC: `0.9931913407537012`
- physical challenge positive rate: `0.8778634202564471`

## Route/Physical Policy Integration

- best mode: `no_route_physical`
- route/physical policy contributes: `False`
- all improvement: `0.18577852429834418`
- t50 improvement: `0.14803699577731477`
- t100 diagnostic improvement: `0.22857426649949408`
- hard/failure improvement: `0.19518047277951456`
- easy degradation: `0.0`
- all delta over no-route-physical: `0.0`
- t50 delta over no-route-physical: `0.0`
- hard delta over no-route-physical: `0.0`

## Route/Physical Group-Consistency Test

- deployable: `True`
- route/physical contributes to group policy: `False`
- all/t50/t100: `0.11568469607260612` / `0.08187483697323228` / `0.12974873518015262`
- hard/failure improvement: `0.12064147528200675`
- easy degradation: `0.0`
- collision delta vs floor @0.05 normalized: `0.00585127847683331`
- all/t50/hard delta vs group consistency: `-0.10671970569760825` / `-0.06906290279898042` / `-0.10347621837197896`

## Joint Route-Conditioned World-State Ablation

- best name: `joint_route_conditioned_ensemble`
- joint route conditioning contributes: `False`
- all improvement: `0.15088774687015682`
- t50 improvement: `0.09295551695088011`
- t100 diagnostic improvement: `0.16374924097478616`
- hard/failure improvement: `0.15655007967571088`
- easy degradation: `0.0`
- all delta over full-trajectory reference: `-0.03489077742818736`
- t50 delta over full-trajectory reference: `-0.05508147882643466`
- t100 delta over full-trajectory reference: `-0.06482502552470792`
- hard delta over full-trajectory reference: `-0.03863039310380367`
- route top1: `0.7032127935455986`
- physical challenge AUROC: `0.9460369245580365`

## Joint Multi-Agent Consistency Calibration

- selected params: `{'mode': 'group_expand', 'expand_risk_max': 0.010222918819636106, 'expand_min_sep': 0.08}`
- joint consistency contributes: `True`
- all improvement: `0.18619055634397086`
- t50 improvement: `0.14841646500755878`
- t100 diagnostic improvement: `0.22857473063742806`
- hard/failure improvement: `0.19563212885213843`
- easy degradation: `0.0`
- all delta over full-trajectory reference: `0.0004120320456266757`
- t50 delta over full-trajectory reference: `0.0003794692302440117`
- hard delta over full-trajectory reference: `0.00045165607262387386`
- expanded-on rows: `118`

## Joint Policy Distillation

- best name: `joint_distill_nobase_balanced::distiller_only`
- joint policy distillation contributes: `True`
- all improvement: `0.28592959855458044`
- t50 improvement: `0.21383787591021597`
- t100 diagnostic improvement: `0.2887528737231674`
- hard/failure improvement: `0.28678460411829854`
- easy degradation: `0.0`
- switch rate: `0.42232747442731594`
- positive external domains: `2`
- all delta over joint consistency: `0.09973904221060959`
- t50 delta over joint consistency: `0.06542141090265718`
- base switch input: `False`
- base-plus-distiller deployable: `False`
- bootstrap all CI low: `0.2816879475496606`
- bootstrap t50 CI low: `0.20700457337312783`
- bootstrap hard/failure CI low: `0.2822097538327134`
- statistically stable on test: `True`
- static causal feature ablation all delta: `-0.17365892957312368`
- full-trajectory signal ablation all delta: `-0.008307576365714331`
- multi-seed pass: `True`
- multi-seed all mean/min: `0.2855577482364627` / `0.2785936928672702`
- multi-seed t50 mean/min: `0.19436183988319766` / `0.1695617463193938`
- multi-seed hard mean: `0.2862145627536274`
- multi-seed easy max: `0.0`

## UCY Fallback Repair

- contributes: `True`
- missing val domains: `['UCY']`
- calibration rows: `9445`
- all improvement: `0.3613141132176878`
- t50 improvement: `0.25956635248380133`
- t100 diagnostic improvement: `0.37474907455985007`
- hard/failure improvement: `0.3616933168487243`
- easy degradation: `0.0`
- UCY all/t50/t100: `0.3928657400363359` / `0.24265047375057225` / `0.4634436152370407`
- all delta over no-UCY policy: `0.07538451466310736`
- t50 delta over no-UCY policy: `0.045728476573585364`
- UCY bootstrap low: `0.38373376338122456`
- train-only UCY calibration: `True`

## UCY Internal Validation

- validation pass: `True`
- source-level independent validation available: `False`
- source-level blocker: `UCY has one train source and no UCY validation source; true source-level UCY validation needs another UCY-like source or a rebuilt split.`
- internal validation pass: `True`
- temporal validation pass: `True`
- test UCY all/t50/t100/hard/easy: `0.3928657400363359` / `0.24265047375057225` / `0.4634436152370407` / `0.3987527223236249` / `0.0`

## Joint Rollout Consistency Audit

- pass: `True`
- policy source: `ucy_repaired_policy`
- all/t50/t100: `0.20681231782543796` / `0.141381936033941` / `0.1391835100380775`
- hard/failure improvement: `0.20016282886383174`
- easy degradation: `0.0`
- multi-agent rows: `54518`
- multi-agent all/t50/hard: `0.20472359322136924` / `0.1425999374065452` / `0.19777460723256124`
- collision delta vs floor @0.05 normalized: `-0.004879122491654175`
- mixed group switch rate: `0.5893493343333959`

## Joint Latent Rollout Prototype

- trained group-token Transformer: `True`
- deployable: `False`
- improves current deployable: `False`
- selected policy: `{'type': 'joint_latent_rollout_val_selected', 'traj_risk_max': -1000000000.0, 'physical_min': 1.1, 'future_close_max': 0.0, 'val_eligible': False}`
- all/t50/t100: `0.0` / `0.0` / `0.0`
- hard/failure improvement: `0.0`
- easy degradation: `0.0`
- raw neural all/t50/hard/easy: `-0.7185960593471277` / `-0.9470536951712247` / `-0.5918476698510013` / `7.0398942498342745`
- all/t50/hard delta vs current group deployable: `-0.139879916889688` / `-0.12149802173622119` / `-0.14504228036236014`
- interaction/occupancy/future-close AUROC: `0.9778962684138552` / `0.9533724454671079` / `0.9718721574984199`

## Joint Residual Rollout Repair

- selected trial: `joint_residual_clip050_safe`
- deployable: `False`
- improves current deployable: `False`
- selected policy: `{'type': 'joint_residual_rollout_val_selected', 'gain_min': 0.4314935505390167, 'harm_max': 8.21683497633785e-05, 'uncertainty_max': 8.21683497633785e-05, 'traj_risk_max': -0.0068512409925460815, 'physical_min': 0.35, 'future_close_max': 0.85, 'val_eligible': True}`
- all/t50/t100: `-0.006199871934993384` / `-0.01622586160854622` / `-0.0055173545259843415`
- hard/failure improvement: `-0.0070840559248055435`
- easy degradation: `0.013581362841401212`
- raw neural all/t50/hard/easy: `-0.3187259758636325` / `-0.4973167857250549` / `-0.27481595341262954` / `3.0795776161376986`
- all/t50/hard delta vs current group deployable: `-0.1460797888246814` / `-0.13772388334476743` / `-0.15212633628716568`
- interaction/occupancy/future-close AUROC: `0.9656852889954971` / `0.9383229818070397` / `0.9677306100575066`

## Joint Residual Domain-Horizon Policy Repair

- selected trial: `joint_residual_clip100_balanced`
- deployable: `False`
- all/t50/t100: `-0.0006363835790781369` / `0.0` / `-0.0005775775254206472`
- hard/failure improvement: `-0.0005725919532908463`
- easy degradation: `0.00111839385467416`
- switch rate: `0.0020530182970753493`
- collision delta @0.05 normalized: `0.0008621005906305768`

## Teacher-Guided Neural Proposal

- selected trial: `teacher_proposal_balanced`
- deployable before repair: `False`
- improves current before repair: `False`
- all/t50/t100: `0.35147372419646705` / `0.23666446707361` / `0.3579659237738704`
- hard/failure improvement: `0.350941376256631`
- easy degradation: `0.0`
- switch rate: `0.46194712577438407`
- collision delta @0.05 normalized: `0.018672731941744014`
- all/t50/t100/hard delta vs current group basis: `0.21159380730677904` / `0.1151664453373888` / `0.189045043847251` / `0.20589909589427088`

## Teacher-Guided Proposal Safety Repair

- deployable after repair: `True`
- improves current after repair: `True`
- selected guard: `{'min_sep': 0.05, 'guarded_off': 10039, 'metrics': {'rows': 53256, 'all_improvement': 0.22280774494692202, 't10_improvement': 0.49519817747255257, 't25_improvement': 0.14144561713770243, 't50_improvement': 0.1747765107374638, 't100_improvement': 0.14077697517497378, 'hard_failure_improvement': 0.21798978349895448, 'easy_degradation': 0.0, 'harm_over_fallback': -0.1073638109520745, 'switch_rate': 0.40692128586450355, 'regret_to_oracle': -0.10582871082615124, 'by_domain': {'ETH_UCY': {'rows': 16103, 'all_improvement': 0.20876672236979266, 't50_improvement': 0.12498790873257482, 't100_improvement': 0.13265281476884816, 'hard_failure_improvement': 0.2049683657193917, 'easy_degradation': 0.0, 'switch_rate': 0.3971309693845867}, 'TrajNet': {'rows': 37153, 'all_improvement': 0.2309271938365859, 't50_improvement': 0.20045724066719228, 't100_improvement': 0.14615779997201717, 'hard_failure_improvement': 0.22622246869978446, 'easy_degradation': 0.0, 'switch_rate': 0.41116464350119775}}}, 'collision_delta_005': -0.009973091470184214, 'switch_rate': 0.40692128586450355, 'eligible': True, 'score': 0.8327436095675232}`
- test guarded off: `9247`
- all/t50/t100: `0.20359710771827477` / `0.13116399043122728` / `0.13371172832175005`
- hard/failure improvement: `0.19657225579495552`
- easy degradation: `0.0`
- switch rate: `0.2954185275896845`
- collision delta @0.05 normalized: `-0.003961994203749264`
- all/t50/t100/hard delta vs current group basis: `0.06371719082858676` / `0.009665968695006091` / `-0.03520915160486934` / `0.05152997543259538`

## Teacher-Guided Repair Bootstrap And Ablation Evidence

- evidence pass: `True`
- bootstrap all/t50/t100/hard lows: `0.19991125433418688` / `0.12503056367179802` / `0.1270625048070677` / `0.19259794747674494`
- bootstrap ETH_UCY/TrajNet/UCY lows: `0.17604451654489478` / `0.21892619124157922` / `0.22065013211805656`
- no-fallback all/easy: `0.296621240422128` / `1.2458611044726973`
- raw policy collision delta @0.05: `0.018672731941744014`
- no-group-consistency all/t50 delta: `-0.1993508302918573` / `-0.13084852602738606`
- no-neighbor-interaction all/t50 delta: `-0.1980910960572272` / `-0.12984928959704223`

## Teacher-Guided Repair Multi-Seed Replication

- replication pass: `True`
- seeds: `[11, 17, 23]`
- all mean/min: `0.20399416929662803` / `0.20358992224459205`
- t50 mean/min: `0.13176918009378483` / `0.13019734119222148`
- t100 mean/min: `0.1349446267607578` / `0.13366944267790382`
- hard mean/min: `0.1970419557530916` / `0.19653792868736553`
- easy max: `0.0`
- collision delta max @0.05: `-0.0037418834146520363`
- positive domain counts: `[3, 3, 3]`

## Composite-Tail Safe-Switch Bounded Neural Dynamics

- evidence pass: `True`
- multiseed pass: `True`
- strict delta vs teacher repair pass: `True`
- all/t50/t100: `0.2102513255185352` / `0.13652231450154184` / `0.14694086716388166`
- hard/failure improvement: `0.20384916307933942`
- easy degradation: `0.0`
- switch rate: `0.3410171445036738`
- collision delta @0.05 normalized: `-0.0038702813749587617`
- bootstrap all/t50/t100/hard lows: `0.20671347297933704` / `0.13060829691569112` / `0.13962817164239194` / `0.19986489207982036`
- delta-vs-teacher CI lows all/t50/t100/hard: `None` / `None` / `None` / `None`
- multiseed means all/t50/t100/hard/easy: `0.20954401723273208` / `0.1383020020634588` / `0.1445226429961963` / `0.203088119625216` / `0.0`
- multiseed delta-vs-teacher mins all/t50/t100/hard: `0.004997329897068581` / `0.005636676602763568` / `0.008330490182942518` / `0.0054317016626029835`
- positive domain counts: `[3, 3, 3]`

## Pure UCY Source-Heldout Validation

- pure UCY source-heldout gate: `True`
- strict pure UCY-only retrain/select/test gate: `False`
- target sources: `['UCY__zara01__obsmat_txt', 'UCY__zara02__obsmat_txt', 'UCY__zara03__crowds_zara03_txt']`
- caveat: `This is a pure-UCY source-heldout frozen-policy check, not a strict pure-UCY-only train/val/test retraining protocol. Coordinates remain dataset-local raw-frame 2.5D.`
- UCY-only policy-head train/val/test gate: `True`
- strict pure UCY-only neural retrain gate: `False`
- UCY policy-head test metrics: `{'rows': 35441, 'all_improvement': 0.20776078287134314, 't10_improvement': 0.3186320645598619, 't25_improvement': 0.07604057090793714, 't50_improvement': 0.16741031165863363, 't100_improvement': 0.219956129029393, 'hard_failure_improvement': 0.20980890977880717, 'easy_degradation': 0.0, 'harm_over_fallback': -0.109476277047378, 'switch_rate': 0.5038232555514799, 'regret_to_oracle': -0.1351880940988382, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': 0.20576698045363073, 't50_improvement': 0.19449331248310975, 't100_improvement': 0.2012278018557263, 'hard_failure_improvement': 0.2074244291310272, 'easy_degradation': 0.0, 'switch_rate': 0.5283965870043628}, 'UCY': {'rows': 9540, 'all_improvement': 0.21306818879480027, 't50_improvement': 0.09380788841505572, 't100_improvement': 0.2737636153082392, 'hard_failure_improvement': 0.21642402352003987, 'easy_degradation': 0.0, 'switch_rate': 0.4371069182389937}}, 'endpoint_fde_metrics': {'rows': 35441, 'all_improvement': 0.19988295321965166, 't10_improvement': 0.34289120068878687, 't25_improvement': 0.07920444370090152, 't50_improvement': 0.1932412042921009, 't100_improvement': 0.19666022159899088, 'hard_failure_improvement': 0.2019465651862199, 'easy_degradation': 0.0, 'harm_over_fallback': -0.15199923397543671, 'switch_rate': 0.5038232555514799, 'regret_to_oracle': 0.055795881982318114, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': 0.200764449256583, 't50_improvement': 0.22997532411686694, 't100_improvement': 0.17668205323818498, 'hard_failure_improvement': 0.20274646874995617, 'easy_degradation': 0.0, 'switch_rate': 0.5283965870043628}, 'UCY': {'rows': 9540, 'all_improvement': 0.19747790428901502, 't50_improvement': 0.08863185042687594, 't100_improvement': 0.2538225381383088, 'hard_failure_improvement': 0.19967489336772315, 'easy_degradation': 0.0, 'switch_rate': 0.4371069182389937}}}, 'collision_delta_vs_floor_005': 0.007327796775769391, 'smoothness_jagged_delta': 0.0, 'alpha_mean': 0.20152930222059195}`
- UCY policy-head caveat: `The policy/head is trained and selected only on UCY train/val rows, but the underlying neural proposal, Stage37 floor, and teacher-repaired switch features are inherited from mixed external training. A full strict pure-UCY neural world-model retrain is still not complete.`

## Domain-Local Neural Endpoint Retrain

- two-domain endpoint gate: `True`
- positive domains: `['ETH_UCY', 'TrajNet', 'UCY_expanded']`
- domain-local caveat: `This is a strict domain-local neural endpoint retrain probe over Stage41 seq2seq causal features. It is not metric, not seconds-level, not true 3D, not foundation, and not a Stage5C/SMC execution.`
- all-agent endpoint-linear proxy gate: `True`
- all-agent proxy positive domains: `['ETH_UCY', 'UCY_expanded']`
- all-agent proxy caveat: `This extends domain-local endpoint-FDE evidence to same-frame all-agent endpoint-linear waypoint safety. It strengthens safety evidence but does not replace a true learned full-trajectory neural dynamics audit.`
- learned full-waypoint domain-local gate: `False`
- learned full-waypoint positive domains: `[]`
- learned full-waypoint failure taxonomy: `{'ETH_UCY': {'reasons': ['t50_ade_not_positive', 't100_ade_not_positive', 'same_frame_proximity_delta_unsafe', 'ungated_neural_catastrophic_easy_degradation'], 'ade_all': 0.03200185360974939, 'ade_t50': 0.0, 'ade_t100': 0.0, 'fde_all': 0.02140015213054336, 'fde_t50': 0.0, 'collision_delta_vs_floor_005': 0.020563149900689304, 'next_fix': 'train a validation-selected efficient proximity guard and horizon-specific waypoint policy; do not deploy learned full-waypoint domain-local model yet.'}, 'TrajNet': {'reasons': ['all_ade_not_positive', 't50_ade_not_positive', 'hard_failure_ade_not_positive', 'same_frame_proximity_delta_unsafe', 'ungated_neural_catastrophic_easy_degradation', 'endpoint_fde_positive_but_waypoint_ade_negative'], 'ade_all': -0.009876103097439914, 'ade_t50': -0.09471803973274517, 'ade_t100': 0.022840260699697912, 'fde_all': 0.06300366873346075, 'fde_t50': 0.06502418472340132, 'collision_delta_vs_floor_005': 0.014909478168264101, 'next_fix': 'train a validation-selected efficient proximity guard and horizon-specific waypoint policy; do not deploy learned full-waypoint domain-local model yet.'}}`
- learned full-waypoint repair two-domain gate: `False`
- learned full-waypoint repair positive domains: `['TrajNet']`
- learned full-waypoint repair summary: `TrajNet t50-only gain-calibrated raw-waypoint policy passed one-domain repair with all=0.00277, t50=0.01093, hard=0.00311, easy=0.00396, collision_delta=-0.00046. ETH_UCY remained failed with t50=0.0 and easy=0.03042. This is a useful neural dynamics slice but not a two-domain deployable neural world model.`
- endpoint-to-full trajectory bridge gate: `True`
- endpoint-to-full positive domains: `['ETH_UCY', 'TrajNet']`
- endpoint-to-full summary: `Endpoint neural dynamics projected through a linear waypoint bridge were scored against reconstructed actual waypoint labels. ETH_UCY all/t50/t100/hard/easy/collision = 0.01570 / 0.00190 / 0.00431 / 0.01547 / 0.0 / -0.00150. TrajNet all/t50/t100/hard/easy/collision = 0.03803 / 0.02648 / 0.01380 / 0.03906 / 0.0 / -0.00291. This is two-domain positive endpoint dynamics evidence, but still not learned full-waypoint shape dynamics.`
- learned waypoint-shape bridge gate: `True`
- learned waypoint-shape positive domains: `['ETH_UCY', 'TrajNet']`
- learned waypoint-shape summary: `A past-only waypoint-shape residual head was trained around the endpoint neural bridge and selected on validation. ETH_UCY all/t50/t100/hard/easy/collision = 0.01570 / 0.00190 / 0.00433 / 0.01548 / 0.0 / -0.00150 with shape gain all/t50/t100/hard = 0.000005 / 0.0 / 0.000014 / 0.000005 and a tiny 0.0046% shape-switch rate. TrajNet all/t50/t100/hard/easy/collision = 0.03824 / 0.02648 / 0.01451 / 0.03930 / 0.0 / -0.00291 with shape gain all/t50/t100/hard = 0.000226 / 0.0 / 0.000718 / 0.000248 and 0.1099% shape-switch rate. This is a protected, tiny, mostly t100/tail learned-shape contribution, not an ungated full-waypoint dynamics replacement.`
- learned shape gain/harm gate: `True`
- learned shape gain/harm summary: `A train-fitted shape gain/harm gate expands ETH_UCY shape intervention and improves TrajNet t50 shape gain, but it is not a monotonic upgrade. ETH_UCY all/t50/t100/hard = 0.01636 / 0.00178 / 0.00427 / 0.01616 with shape-gain delta vs previous all/t100 = +0.000674 / -0.000059. TrajNet all/t50/t100/hard = 0.03814 / 0.02693 / 0.01380 / 0.03918 with shape-gain delta vs previous all/t100 = -0.000109 / -0.000718. This points to a domain/horizon-specific shape-policy composer rather than a single global shape gate.`
- domain/horizon shape-policy composer: `True`
- domain/horizon shape-policy composer summary: `A val-selected composer chooses bridge / previous learned-shape / gain-gate sources separately for short, t50, and t100 horizons. ETH_UCY selected gain_gate / bridge / old_shape and scored all/t50/t100/hard/easy = 0.01641 / 0.00190 / 0.00433 / 0.01621 / 0.0. TrajNet selected bridge / gain_gate / gain_gate and scored all/t50/t100/hard/easy = 0.03814 / 0.02693 / 0.01380 / 0.03918 / 0.0. This formalizes the mixed shape evidence as a protected horizon policy, but it is still not an ungated full-waypoint neural dynamics replacement.`
- dynamic shape source meta-policy: `True`
- dynamic shape source meta-policy summary: `A per-row expected-ADE source model over bridge / old-shape / gain-gate sources passes a protected two-domain gate. ETH_UCY all/t50/t100/hard/easy = 0.01631 / 0.00178 / 0.00428 / 0.01611 / 0.0, but its test source-ranking accuracy is only 0.0193 and it is weaker than the fixed composer. TrajNet all/t50/t100/hard/easy = 0.03830 / 0.02672 / 0.01451 / 0.03936 / 0.0, improving fixed composer on all/hard/t100 but not t50. Dynamic source selection is useful but not yet the deployable default over the fixed composer.`
- calibrated shape source meta-policy: `diagnostic_negative`
- calibrated shape source meta-policy summary: `Validation-only affine log calibration was tested globally, per source, and per source+horizon to repair ETH_UCY source ranking. Validation selected no calibration for both ETH_UCY and TrajNet. ETH_UCY remained weaker than fixed composer with rank accuracy 0.0193 and fixed deltas all/t50/t100/hard = -0.000099 / -0.000125 / -0.000044 / -0.000095. This rules out simple raw-ADE calibration as the repair path; the next fix should use pairwise gain/harm source switching.`
- pairwise gain/harm shape switch policy: `diagnostic_safe_not_deployable_over_fixed`
- pairwise gain/harm shape switch policy summary: `A pairwise source-switch model predicts gain and harm for switching from the protected bridge into old-shape or gain-gate sources. It passes the two-domain positive/safety gate, but ETH_UCY is weaker than fixed composer and TrajNet only improves all/hard/t100 by tiny amounts while t50 falls. Current deployable shape policy remains the fixed horizon composer/composite route; pairwise switching is diagnostic evidence for future source-specific hard/failure features.`
- weighted pairwise shape switch policy: `diagnostic_safe_not_deployable_over_fixed`
- weighted pairwise shape switch policy summary: `Hard/failure, t50/t100, source-switch, and positive-gain rows were upweighted during pairwise training. This did not repair the deployability gap: ETH_UCY remains below fixed composer and TrajNet t50 remains below fixed composer. The blocker is likely source-specific causal feature insufficiency or missing per-domain/t50 priors rather than simple positive-label imbalance.`
- fixed-composer prior source switch policy: `diagnostic_negative_not_deployable_over_fixed`
- fixed-composer prior source switch policy summary: `The fixed composer was treated as the safety prior and the learned model was only allowed to make residual source switches after validation proved t50 preservation and all/hard/t100 gain. The stricter policy prevented TrajNet t50 harm by falling back to the fixed composer, but it did not beat fixed on either domain: ETH_UCY deltas all/t50/t100/hard = -0.000075 / 0.0 / -0.000044 / -0.000077, while TrajNet deltas are exactly 0. Current deployable shape path remains the fixed composer/composite route.`

## Neural Group Consistency Distiller

- deployable: `True`
- improves fixed proximity guard: `True`
- selected policy: `{'type': 'group_consistency_distiller', 'safe_min': 0.4, 'gain_min': 0.25296418368816376, 'unsafe_max': 0.8, 'val_eligible': True}`
- all/t50/t100: `0.22240440177021437` / `0.1509377397722127` / `0.23019369783249866`
- hard/failure improvement: `0.2241176936539857`
- easy degradation: `0.0`
- switch rate: `0.265541708687509`
- collision delta vs floor @0.05 normalized: `0.00829083972266037`
- lift over fixed guard all/t50/t100/hard: `0.015592083944776403` / `0.009555803738271695` / `0.09101018779442116` / `0.023954864790153962`
- bootstrap all/t50/t100/hard CI lows: `0.2185104674424955` / `0.1445060231000635` / `0.22247098030108228` / `0.2197743039844412`
- statistically stable on test: `True`
- group-consistency feature ablation all/t100 delta: `-0.22219845243926872` / `-0.23019369783249866`
- proposal-score feature ablation all delta: `-0.22195167063440968`

## Conclusion

M3W-Neural v1 is now more than an endpoint-only candidate: the fresh full-trajectory probe adds waypoint trajectory, interaction-risk, occupancy, and physical-validity heads, and the goal/route repair pass adds an explicit route head plus a non-degenerate physical-challenge target. The route/physical heads are useful diagnostics, but post-hoc route/physical gating, joint route-conditioned training, and route/physical-augmented group consistency are negative ablations for trajectory deployment, so route/physical is diagnostic-only in the current deployable path. Joint policy distillation learns gain/harm/switch without base-switch input and is statistically stable across bootstrap plus three seeds. The UCY fallback-only blocker was traced to missing UCY validation rows and repaired with train-only UCY calibration. A neural group-consistency distiller improves the fixed joint proximity guard, and a validation-selected safety-buffer repair passes all three seeds while preserving easy cases and joint proximity safety. A teacher-guided neural proposal then uses train-only teacher switch labels and neural proposal scores to exceed the group-consistency safety-buffer basis on all/t50/hard; its raw proposal was unsafe, but a validation-selected proximity repair restores joint safety while retaining positive all/t50/hard lift. The newer composite-tail safe-switch bounded neural dynamics candidate keeps easy=0, has positive bootstrap CI lows, passes three seed-aware evaluations, improves the teacher repair on all/t50/t100/hard, and is positive on pure-UCY source-heldout checks. The fresh all-agent composite world-state audit applies that frozen composite-tail policy to full future waypoint rollouts for every active row in same-frame groups; it passes ADE/FDE, multi-agent, proximity, and smoothness gates under the safety floor. No-fallback/full-row neural remains unsafe for easy cases, so the Stage37/teacher safety floor remains required. A fresh joint latent group-token rollout prototype learned strong interaction/occupancy/future-close auxiliary signals but raw neural rollout was FDE-negative, so the validation policy selected fallback-only and the prototype is not deployable. Baseline-relative bounded residual rollout reduced raw neural damage but still failed all/t50/hard gates, and the domain/horizon residual repair still did not produce positive all/t50/hard transfer. JEPA is formally disabled from the deployable path because audited non-collapse JEPA variants did not produce deployable downstream lift. This remains protected grouped 2.5D rollout evidence rather than latent generative world-state execution. The M3W-Neural v1 audit matrix is complete for the current protected candidate, while stricter pure UCY-only retrain/select/test and ungated full-row neural safety remain future-strengthening work; Stage5C/SMC stay disabled.
