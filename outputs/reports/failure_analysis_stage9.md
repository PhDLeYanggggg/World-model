# Stage 9 Failure Analysis

Full model all-test mean improvement: -0.001592. This does not meet the 5% gate.
Hard/failure best improvement: 0.000537. This does not meet the 10% gate.
Interaction gain over scene+goal: -0.000414. Interaction is not proven useful for trajectory metrics.
Scene/goal gain over no-scene: -0.000005. Scene/goal grounding is not yet producing reliable trajectory lift.
The strongest causal baselines remain very hard to beat on t+10 pedestrian-like snippets.
No verified pedestrian/drone t+50/t+100 is available, so no long-horizon pedestrian world-model claim is allowed.
