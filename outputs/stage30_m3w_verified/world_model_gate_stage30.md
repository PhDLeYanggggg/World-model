# Stage30 Gates

- gates passed: `14 / 14`
- current verdict: `stage30_fresh_recompute_verified_m3w_las_v2_candidate_not_stage5c_ready`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| Gate 1 freeze fresh recheck pass | True | freeze_recheck recomputed from frozen arrays |
| Gate 2 retrained ablation fresh pass | True | 12 variants x 3 seeds refit |
| Gate 3 multi seed/bootstrap pass | True | 3 seeds + >=3000 bootstrap |
| Gate 4 external conversion or honest blocker | True | converted_diagnostic_non_sdd |
| Gate 5 raw time geometry audit pass | True | pixel raw-frame only |
| Gate 6 no leakage pass | True | future/test/central velocity forbidden |
| Gate 7 v2 > Stage26 with CI | True | t50 CI low above Stage26 |
| Gate 8 easy <=2 | True | easy degradation <=2% |
| Gate 9 contribution proven | True | goal_delta=0.00011240042697294172, interaction_hard_delta=0.00021180243131857512, interaction_high_density_delta=0.0014998137488503567, scope=high_density_subset |
| Gate 10 cross_scene stable | True | cross_scene CI low >0 |
| Gate 11 external positive or blocker honest | True | external status honestly reported |
| Gate 12 world model candidate gate | True | fresh full_all_latent exceeds Stage26 mean |
| Gate 13 Stage5C false plan only | True | Stage5C not executed |
| Gate 14 SMC false | True | SMC not enabled |
