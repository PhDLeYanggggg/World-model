# Stage 23 Gates

- gates passed: `5 / 14`
- Stage 5C readiness: `False`
- SMC readiness: `False`
- quick-plus is explicitly not full medium.

| gate | pass | evidence |
| --- | --- | --- |
| Gate 1: Medium Data Gate | False | Full medium episodes required; quick-plus is partial. |
| Gate 2: Dual Split Gate | True | cross_scene and within_scene splits built and audited. |
| Gate 3: Time/Geometry Gate | True | Conclusion: pixel-space only, effective seconds unknown. |
| Gate 4: Strong Baseline Gate | True | Medium/quick-plus strongest baselines computed. |
| Gate 5: Hard/Failure Gate | True | HardBench and BaselineFailureBench enough. |
| Gate 6: GoalBench Gate | True | within_scene GoalBench meaningful; cross_scene goals diagnostic. |
| Gate 7: Selector Gate | False | Validation-selected selector must improve >=5%. |
| Gate 8: Failure Predictor Gate | False | Failure predictor AUROC >=0.75 required. |
| Gate 9: JEPA Gate | False | JEPA non-collapse plus downstream lift required. |
| Gate 10: Correction Gate | False | Correction must improve hard/failure without easy degradation. |
| Gate 11: Scene/Goal Gate | False | Scene/goal lift not demonstrated. |
| Gate 12: Interaction Gate | False | Interaction lift not demonstrated. |
| Gate 13: Stage 5C Readiness Gate | False | Keep false; do not execute Stage 5C. |
| Gate 14: SMC Readiness Gate | False | Keep false. |
