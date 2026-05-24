# M3W State Machine Stage C Gates

- gates passed: `5 / 5`
- current verdict: `stage_c_pass_enter_stage_d`
- Stage5C execution: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| C1 All-Latent Beats Stage26 | True | all_latent exceeds Stage26 on t+50 or hard/failure |
| C2 Hard/Failure >= 10% | True | 0.1336398986813968 >= 0.10 |
| C3 Easy <= 2% | True | 0.01928694490688554 <= 0.02 |
| C4 JEPA or Transformer Downstream Lift | True | no-JEPA/no-Transformer retrained variants improve over Stage26-only |
| C5 Goal or Interaction Contribution | True | goal or interaction ablation reduces performance |
