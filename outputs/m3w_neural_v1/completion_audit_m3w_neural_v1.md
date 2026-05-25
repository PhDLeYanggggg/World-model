# M3W-Neural v1 Completion Audit

- source: `fresh_run`
- completion_status: `not_complete`
- current_best_deployable: `M3W-Neural v1 self-gated endpoint candidate under Stage37 safety floor`

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

## Conclusion

M3W-Neural v1 is now more than an endpoint-only candidate: the fresh full-trajectory probe adds waypoint trajectory, interaction-risk, occupancy, and physical-validity heads, and the goal/route repair pass adds an explicit route head plus a non-degenerate physical-challenge target. The full active objective is still not complete because the rollout is still per-agent all-agent-context rather than a jointly consistent latent world-state model, and route/physical heads have not yet been shown to improve the trajectory deployment policy.
