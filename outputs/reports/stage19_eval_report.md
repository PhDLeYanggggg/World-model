# Stage 19 Evaluation Report

1. simulation pretraining 是否提升 real failure predictor？`False`
2. egocentric video pretraining 是否提升 scene/goal/failure representation？`False`
3. top-down video/raster 是否提升 baseline selector？`True`
4. JEPA 是否提升 official t+50？`True`
5. JEPA 是否提升 hard/failure？`False`
6. JEPA 是否改善 Stage17 selector？`True`
7. JEPA 是否改善 Stage18 failure predictor？`True`
8. JEPA 是否产生 collapse？`False`
9. 是否仍需要 SDD/OpenTraj 本地数据？`True`

- failure AUROC no-JEPA: `0.647172`
- failure AUROC WAM-JEPA: `0.676029`
- selector t+50 no-JEPA: `0.081954`
- selector t+50 WAM-JEPA: `0.082243`
- hard/failure FDE improvement: `0.000000`
- official t+50 FDE improvement over Stage17: `0.000289`
- sim-to-real gap: `must be measured; simulation metrics are not real-world success`
