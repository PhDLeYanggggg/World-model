# Stage30 B Fresh Retrained Ablation

- source: `fresh_run` for all rows.
- This refits selectors with seeds 0/1/2; it does not read Stage28 ablation as new evidence.
- cached inputs: Stage26 feature store and Stage28 latent cache are hash-verified inputs, not fresh results.
- caveat: no_scene/no_goal/no_interaction drop Stage26 feature groups; frozen latents may still contain mixed information.

| variant | t50 mean | t50 std | hard mean | easy mean | regret mean | switch mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| full_all_latent | 0.159624 | 0.000100 | 0.127753 | 0.017766 | 3.555757 | 0.050000 |
| no_jepa | 0.152512 | 0.000565 | 0.126241 | 0.017930 | 3.594969 | 0.050000 |
| no_transformer | 0.158711 | 0.000667 | 0.128312 | 0.017986 | 3.541796 | 0.050000 |
| no_scene | 0.159666 | 0.000278 | 0.128455 | 0.017374 | 3.537702 | 0.050000 |
| no_goal | 0.159511 | 0.000262 | 0.127803 | 0.017517 | 3.554517 | 0.050000 |
| no_interaction | 0.159394 | 0.000693 | 0.127542 | 0.017469 | 3.560964 | 0.050000 |
| no_failure_hidden | 0.159324 | 0.000278 | 0.127590 | 0.017779 | 3.559952 | 0.050000 |
| no_simulation_curriculum | 0.159624 | 0.000100 | 0.127753 | 0.017766 | 3.555757 | 0.050000 |
| no_fallback | 0.195870 | 0.000773 | 0.163389 | 0.052655 | 2.615873 | 0.714913 |
| no_stage26_features | 0.159269 | 0.001008 | 0.128614 | 0.018259 | 3.533629 | 0.050000 |
| latent_only | 0.158848 | 0.000934 | 0.128387 | 0.018002 | 3.539465 | 0.050000 |
| stage26_only | 0.100135 | 0.000699 | 0.079798 | 0.044128 | 4.789405 | 0.050000 |
