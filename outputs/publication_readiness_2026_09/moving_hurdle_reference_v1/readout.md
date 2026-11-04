# Source-excluded reference prediction

Fresh216forests /9216primary trees, three actual seeds, direct64versus hurdle32+32trees. No trajectory model, policy or threshold changes. All108OOF arrays and216fit bindings replay; all18held prediction sets (108arrays) and metrics replay exactly.

|Fixed H contrast; positive is lower loss|Normalized loss reduction [grouped95%CI]|
|---|---:|
|direct_minus_prior_MSE_reduction|+7.07573e-05 [+1.05248e-05,+0.000149199]|
|hurdle_minus_prior_MSE_reduction|+4.86479e-05 [-5.94205e-06,+0.000113995]|
|hurdle_minus_direct_MSE_reduction|-2.21094e-05 [-4.30839e-05,-3.67567e-06]|
|C_selected_minus_prior_MSE_reduction|+6.88075e-05 [+9.50482e-06,+0.000147169]|
|event_minus_prior_Brier_reduction|+0.00043261 [-0.010601,+0.0109385]|

C-only source-OOF selections:{"direct": 16, "hurdle": 2, "prior": 0}. Positive C-selected reference prediction effect:True; positive hurdle-versus-direct effect:False; positive event Brier effect:False.

The intervals concern normalized ER prediction MSE or event Brier, NOT ADE/FDE percentages, trajectory utility, physical safety or independent risk calibration. All23supported source groups are exposed development; original24primary stays not_estimable. Null source AUCs remain not_estimable. Point prediction can improve while a finite selection still has no supporting event. No conditional independence of agents, no record-event guarantee and no missing-at-random assumption.

Unknown-label rows remain in inference but outside this supervised conditional-known-label loss; this head cannot simply replace the original conservative bounded-reference decoder without accounting for unknown label availability. Positive conditional heads use the original loss measure restricted to positive events, not a second source/query rebalance.

All model selection is complete source-excluded internal C validation. Frozen producer forecasts exclude every C/H source; no C-trained critic or global C normalizer is an input. Inputs are raw local380past/current/predicted-rollout descriptors, logquerysize and logcausalscale. Future masks and endpoints are labels only. No new official data roles or reserved-source opening.

Private weights/OOF arrays are not committed. See per-source results, exact provenance verification, runtime receipts and operation guide. No metric/seconds/true3D/foundation/human-gold claim, Stage5C or SMC.
