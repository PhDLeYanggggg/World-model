# Stage40 Gates

- gates passed: `11 / 12`
- verdict: `stage40_neural_optimization_keep_stage37`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| Gate1 Stage39 failure diagnosis complete | True | {'transformer': 'fallback gate selected no reliable neural switches; same-subset metrics equal Stage37, so Transformer learned no deployable dynamics lift', 'jepa': 'non-collapse but downstream failure AUROC lift is negative; representation adds noise rather than useful selector/failure signal', 'hybrid': 'JEPA auxiliary does not improve dynamics; hybrid remains below or equal to Stage37 under safety gate', 'without_fallback': 'unprotected endpoint dynamics is not competitive with strong causal/Stage37 floor and would risk easy degradation', 'objective': 'raw endpoint/FDE loss does not explicitly teach Stage37 switch/harm/gain mechanism', 'domain': 'UCY-only held-out means ETH/TrajNet external evidence remains blocked'} |
| Gate2 rebuilt Stage37-supervised objectives | True | candidate rel-FDE, oracle/teacher margin, switch/gain/harm, t50/hard weighting |
| Gate3 at least three neural model classes tried | True | ['causal_transformer_candidate_ranker', 'hard_failure_oversampled_ranker', 'hybrid_moe_deeper_ranker', 'jepa_aux_candidate_ranker', 'stage37_teacher_distilled_safe_ranker', 't50_curriculum_ranker'] |
| Gate4 bounded optimization loop executed | True | 6 |
| Gate5 neural_with_fallback beats Stage37 on all/t50/hard | False | {'rows': 16000, 'all_improvement': 0.13200527968500975, 't10_improvement': 0.3025072845476737, 't25_improvement': 0.0, 't50_improvement': 0.08301228991324938, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1523895057047563, 'easy_degradation': 0.0006815189764808327, 'harm_over_fallback': -0.13887056636996567, 'switch_rate': 0.0, 'neural_without_fallback': {'rows': 16000, 'all_improvement': -1.2636329485667623, 't10_improvement': -0.6609946384475607, 't25_improvement': -1.7309633497217503, 't50_improvement': -2.921009455663605, 't100_improvement': -0.6466884359881155, 'hard_failure_improvement': -1.093965878739179, 'easy_degradation': 6.123116549641697, 'harm_over_fallback': 1.3293515507103102}} |
| Gate6 easy degradation <=2 | True | 0.0006815189764808327 |
| Gate7 SDD safety not destroyed | True | preserved_by_stage37_fallback_or_no_deployment |
| Gate8 no leakage pass | True | {'future_endpoint_input': False, 'future_labels_for_loss_only': True, 'central_velocity': False, 'test_endpoint_goals': False} |
| Gate9 t100 diagnostic honest | True | 0.0 |
| Gate10 ETH/TrajNet/OpenTraj repaired or blocker | True | {'UCY': 'available_heldout_test', 'ETH_UCY': 'not_run_blocker: available rows are train-only under frozen Stage37 split; rebuilding held-out test would invalidate frozen policy/test protocol', 'TrajNet': 'not_run_blocker: train/val rows exist but no frozen held-out test split; requires Stage40 split rebuild and retuning on val only', 'OpenTraj_mixed': 'not_run_blocker: mixed test currently UCY; non-UCY held-out requires new split'} |
| Gate11 Stage5C false | True | Stage5C not executed |
| Gate12 SMC false | True | SMC not enabled |
