# Stage 17 Failure Analysis

- Oracle baseline selection has headroom, meaning the candidate baseline family can beat the global baseline when future labels choose the best member.
- The trained causal selector only recovers part of that headroom and does not pass the >=3% selector gate.
- Correction specialist has no reliable incremental gain over selector-only without risking easy degradation.
- Scene/goal features remain unproven; interaction contribution is at most diagnostic.
- BPSG-MA v1 remains the deployable model.
