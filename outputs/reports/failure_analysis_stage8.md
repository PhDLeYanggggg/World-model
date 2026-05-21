# Stage 8 Failure Analysis

The largest failures are data-grounding failures, not a lack of larger neural architecture.

- No local SDD/OpenTraj long-horizon pedestrian/drone source was verified during this run.
- Gold/silver scene annotations: 0; most scene goals remain inferred-only.
- Goal predictor test metrics: {'samples': 12, 'top1_accuracy': 0.5, 'top3_accuracy': 0.75, 'goal_NLL': 1.255832, 'goal_ECE': 0.104385, 'goal_entropy': 1.245901, 'majority_top1': 0.333333, 'majority_top3': 0.666667, 'hard_failure_goal_accuracy': 0.5, 'beats_majority': True, 'top3_saturated': False}
- Best BaselineFailureBench improvement: -0.001179
- Best HardBench improvement: 0.000266
- The residual head corrects only the primary agent in the multi-agent episode, so it is not yet a full multi-agent dynamics model.
- Do not enter Stage 5C until deterministic scene/goal correction passes the failure/hard gates.
