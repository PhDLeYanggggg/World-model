# Whole-Fitting All/Easy Task Attribution

Fresh diagnostic, frozen neural models. Full empirical query-draw weights,
known labels plus unchanged unknown-outcome bounds; only fitting B.
Nine dependent checkpoint views and 27 disposable saved-AdamW steps.
No persisted weights, changed policy, new V/H or independent test.

| Context | Model | Shared all/easy cosine | Shared joint/easy cosine | Easy MSE before | Joint step | Protected step | Easy-only step |
|---|---|---:|---:|---:|---:|---:|---:|
| eqmotion_fold1_seed17_critic0 | warm | 0.9672928506984803 | 0.9989871356079446 | 0.0337403406 | 0.034003272 | 0.034003272 | 0.0338371586 |
| eqmotion_fold1_seed17_critic0 | unmasked_MSE | 0.9825692228376851 | 0.9994627635390959 | 0.0197207471 | 0.0197222952 | 0.0197222952 | 0.0195007676 |
| eqmotion_fold1_seed17_critic0 | scale_ablated_MSE | 0.9967553506984502 | 0.9999086535738348 | 0.0295159842 | 0.0295512565 | 0.0295512565 | 0.0288430001 |
| eqmotion_fold0_seed17_critic1 | warm | 0.9202240159879702 | 0.9994640898045808 | 0.0326196256 | 0.0338881356 | 0.0338881356 | 0.0341971969 |
| eqmotion_fold0_seed17_critic1 | unmasked_MSE | 0.41499105205852443 | 0.9985692111072642 | 0.0191374716 | 0.020649734 | 0.020649734 | 0.020693239 |
| eqmotion_fold0_seed17_critic1 | scale_ablated_MSE | 0.6381719999660054 | 0.9911999395519718 | 0.0182351735 | 0.0182710696 | 0.0182710696 | 0.0182746257 |
| eqmotion_fold0_seed17_critic2 | warm | 0.7618595588346568 | 0.9957845587756297 | 1.54160975 | 1.52763557 | 1.52763557 | 1.52952098 |
| eqmotion_fold0_seed17_critic2 | unmasked_MSE | 0.8134309306320547 | 0.9959250767897492 | 1.16182706 | 1.16094006 | 1.16094006 | 1.16207582 |
| eqmotion_fold0_seed17_critic2 | scale_ablated_MSE | 0.7752373288305978 | 0.9950950630843041 | 1.18013923 | 1.18060265 | 1.18060265 | 1.18175917 |

Original MSE joint first-order easy-harm contexts: 0/3.
Actual projected-step easy MSE better: 0/3.
Projected-gradient followup supported: False.

The counterfactual retains AdamW momentum and clipping, so raw gradient
cosine alone is not the actual optimizer effect. A local fitting step
does not prove the cause of the complete training path or held transfer.
The old European membership-BCE/total-cost audit studied a different
task pair and did not support a broad conflict claim; its failure remains.
Original 2% rules and support unchanged. Pixel obs8/pred12, not metric,
seconds, true 3D, foundation or human gold. Stage5C/SMC off;
submission_ready remains false.
