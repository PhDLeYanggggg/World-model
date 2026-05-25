# M3W-Neural v1 Completion Audit

- source: `fresh_run`
- completion_status: `not_complete`
- current_best_deployable: `M3W-Neural v1 group-consistency-distilled joint-safe candidate under Stage37 safety floor`

## Requirement Matrix

| Requirement | Status | Evidence | Note |
| --- | --- | --- | --- |
| external split covers ETH/UCY/TrajNet or blockers | `complete` | outputs/stage41_external_split/report.json and Stage41 gates |  |
| no leakage: future endpoint label/eval only, no central velocity, no test endpoint goals | `complete` | outputs/stage41_breakthrough/stage41_endpoint_geometry_audit.json |  |
| neural model exceeds Stage37 on external all/t50/hard with easy <=2 | `complete` | outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json |  |
| at least two held-out external domains positive | `complete` | outputs/stage41_breakthrough/stage41_neural_eval.json |  |
| neural without external fallback not catastrophic | `complete` | fresh self-gated endpoint records no-external-fallback safe, but raw ungated endpoint remains unsafe in Stage41 reports | The self-gated neural output is safe; raw ungated endpoint dynamics remain unsafe and are not deployable. |
| all active agents future world-state, not only endpoint selector | `partial` | outputs/stage41_breakthrough/stage41_all_agent_eval.json, stage41_all_agent_risk_repair.json, stage41_all_agent_t50_specialist.json, stage41_all_agent_policy_composer.json, outputs/stage41_stratified_protocol/stage41_fixed_policy_confirmation.json, outputs/stage41_fresh_confirmation/stage41_fresh_all_agent_endpoint_specialist.json, and outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json | Fresh full-trajectory probe reconstructs actual future waypoint labels from raw external trajectories and trains trajectory, interaction-risk, occupancy, and physical-validity heads with positive ETH_UCY/TrajNet transfer. It remains per-agent all-agent-context prediction with goal/route proxy features, not a fully joint latent world-state rollout, so the full objective remains not complete. |
| full trajectory, interaction, occupancy, and physical-validity heads | `complete` | outputs/stage41_fresh_confirmation/stage41_full_trajectory_world_state.json | Trajectory ADE/t50/t100/hard improve with easy preserved; interaction and occupancy heads report AUROC/AUPRC. The separate goal/route/physical repair pass adds a non-degenerate physical-challenge label. |
| explicit goal/route head and non-degenerate physical-consistency target | `complete` | outputs/stage41_fresh_confirmation/stage41_goal_route_physical_repair.json | Route top1 beats majority and physical-challenge AUROC is high. Labels are still supervised future-waypoint targets, never inference inputs. |
| route/physical heads improve trajectory deployment policy | `partial` | outputs/stage41_fresh_confirmation/stage41_route_physical_policy_integration.json and outputs/stage41_fresh_confirmation/stage41_joint_route_conditioned_world_state.json | Auxiliary route/physical heads are predictive diagnostics, but post-hoc route/physical gating selected the no-route-physical policy and joint route-conditioned trajectory training underperformed the full-trajectory reference. |
| joint multi-agent consistency improves trajectory deployment policy | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_multiagent_consistency.json | Current-frame group consistency adds a tiny positive deployment-policy lift over the full-trajectory reference and gives UCY a small positive switch rate, but it is still post-hoc group calibration rather than a jointly consistent latent rollout. |
| neural gain/harm/switch distillation improves deployment without base-switch leakage | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation.json | The deployable distiller-only policy learns gain/harm/switch from train labels and uses past/static/full-trajectory prediction signals at inference. It improves ETH_UCY and TrajNet but still falls back on UCY. |
| no-base-switch distiller bootstrap and ablation evidence | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_evidence.json | Bootstrap lower bounds are positive for all/t50/hard; ablations show static causal features and full-trajectory prediction signals are the main positive contributors, while UCY remains fallback-only. |
| no-base-switch distiller multi-seed replication | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_policy_distillation_multiseed.json | Three fresh seeds keep all/t50/t100/hard positive with easy preserved and two positive domains per seed; UCY remains fallback-only. |
| UCY fallback-only blocker diagnosed and repaired without test tuning | `complete` | outputs/stage41_fresh_confirmation/stage41_ucy_fallback_repair.json | UCY was missing from validation, so no UCY slice thresholds were selected. A train-only UCY calibration subset repairs UCY on test, but independent UCY validation is still needed before final deployment. |
| UCY repair internal fold/temporal validation | `complete` | outputs/stage41_fresh_confirmation/stage41_ucy_independent_validation.json | UCY repair validates on internal held-out row folds and temporal blocks. True source-level UCY validation remains unavailable because there is one UCY train source and no UCY validation source. |
| grouped all-agent rollout consistency under repaired policy | `complete` | outputs/stage41_fresh_confirmation/stage41_joint_rollout_consistency.json | Audits same-frame multi-agent selected future waypoints for switch coherence, proximity risk, smoothness, and multi-agent improvement. This is grouped rollout evidence, not Stage5C latent generation or SMC. |
| neural group-consistency head improves joint-safe fixed proximity guard | `complete` | outputs/stage41_fresh_confirmation/stage41_group_consistency_distiller.json | Trains a neural safe-switch/gain/unsafe head from train labels and selects thresholds on validation. It improves the fixed proximity guard while preserving easy cases and joint proximity safety, but it is still a guarded selector/dynamics head rather than Stage5C latent generation. |
| t100 diagnostic positive or blocker analysis | `complete` | outputs/m3w_neural_v1/evidence_matrix_m3w_neural_v1.json |  |
| JEPA contribution proven or disabled | `partial` | Stage41 final report: JEPA not proven unless winning trial passes; winning frozen candidate is self-gated endpoint dynamics, not JEPA contribution. |  |
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

## Conclusion

M3W-Neural v1 is now more than an endpoint-only candidate: the fresh full-trajectory probe adds waypoint trajectory, interaction-risk, occupancy, and physical-validity heads, and the goal/route repair pass adds an explicit route head plus a non-degenerate physical-challenge target. The route/physical heads are useful diagnostics, but post-hoc route/physical gating and joint route-conditioned training are negative ablations for trajectory deployment. Joint policy distillation learns gain/harm/switch without base-switch input and is now statistically stable across bootstrap plus three seeds. The UCY fallback-only blocker was traced to missing UCY validation rows and repaired with train-only UCY calibration. A neural group-consistency distiller now improves the fixed joint proximity guard while preserving easy cases and joint proximity safety. This remains grouped 2.5D rollout evidence rather than latent generative world-state execution. The full active objective is still not complete because source-level independent UCY validation remains unavailable and Stage5C/SMC stay disabled.
