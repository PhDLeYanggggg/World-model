# Stage28 Retrained Ablation Table

- Each row retrains a selector variant on train and selects fallback policy on validation.
- SDD remains pixel-space raw-frame; no metric/seconds claim.

| ablation | variant | model | t50 | hard/failure | easy degradation | regret | switch rate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| stage26_only | stage26_only | extra_trees | 0.125127 | 0.115358 | 0.018596 | 3.872726 | 0.043350 |
| no_jepa | plus_transformer | ridge | 0.162020 | 0.125941 | 0.020026 | 3.602788 | 0.059300 |
| no_transformer | plus_jepa | ridge | 0.166755 | 0.131590 | 0.020609 | 3.456247 | 0.058520 |
| no_scene | all_latent | extra_trees | 0.169323 | 0.134278 | 0.019456 | 3.388692 | 0.050000 |
| no_goal | all_latent | extra_trees | 0.156723 | 0.120642 | 0.015613 | 3.739771 | 0.039840 |
| no_interaction | all_latent | extra_trees | 0.167504 | 0.133221 | 0.017313 | 3.415816 | 0.050000 |
| no_failure_hidden | plus_hybrid | ridge | 0.160555 | 0.131819 | 0.019736 | 3.449150 | 0.059830 |
| no_simulation_curriculum | all_latent | ridge | 0.158434 | 0.120730 | 0.013016 | 3.737507 | 0.050000 |
| no_fallback | all_latent | extra_trees | 0.194448 | 0.166825 | 0.050356 | 2.528794 | 0.742310 |
