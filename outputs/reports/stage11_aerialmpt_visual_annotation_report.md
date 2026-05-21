# Stage 11 AerialMPT Visual Annotation Report

Dataset: `aerialmpt`
License: `CC BY-SA 4.0`
AI visual silver scenes: `14`
Metric status: `pixel_only_no_homography`

| scene | split | frames | agents | detections | t+10 | preview |
| --- | --- | --- | --- | --- | --- | --- |
| bauma1 | train | 19 | 270 | 4448 | False | outputs/figures/stage11_visual_annotations/aerialmpt_bauma1_visual_silver.png |
| bauma2 | train | 29 | 148 | 3627 | True | outputs/figures/stage11_visual_annotations/aerialmpt_bauma2_visual_silver.png |
| bauma3 | test | 16 | 609 | 8788 | False | outputs/figures/stage11_visual_annotations/aerialmpt_bauma3_visual_silver.png |
| bauma4 | train | 22 | 127 | 2399 | True | outputs/figures/stage11_visual_annotations/aerialmpt_bauma4_visual_silver.png |
| bauma5 | train | 17 | 94 | 1410 | False | outputs/figures/stage11_visual_annotations/aerialmpt_bauma5_visual_silver.png |
| bauma6 | test | 26 | 270 | 5314 | True | outputs/figures/stage11_visual_annotations/aerialmpt_bauma6_visual_silver.png |
| karlsplatz | test | 27 | 146 | 3374 | True | outputs/figures/stage11_visual_annotations/aerialmpt_karlsplatz_visual_silver.png |
| marienplatz | train | 30 | 215 | 5158 | True | outputs/figures/stage11_visual_annotations/aerialmpt_marienplatz_visual_silver.png |
| oac | train | 18 | 92 | 1287 | False | outputs/figures/stage11_visual_annotations/aerialmpt_oac_visual_silver.png |
| pasing1L | train | 28 | 100 | 2327 | True | outputs/figures/stage11_visual_annotations/aerialmpt_pasing1L_visual_silver.png |
| pasing1R | train | 16 | 86 | 1196 | False | outputs/figures/stage11_visual_annotations/aerialmpt_pasing1R_visual_silver.png |
| pasing7 | test | 24 | 103 | 2064 | True | outputs/figures/stage11_visual_annotations/aerialmpt_pasing7_visual_silver.png |
| pasing8 | test | 27 | 83 | 1932 | True | outputs/figures/stage11_visual_annotations/aerialmpt_pasing8_visual_silver.png |
| witt | test | 8 | 185 | 1416 | False | outputs/figures/stage11_visual_annotations/aerialmpt_witt_visual_silver.png |

## Limitations

- No homography or meter scale is available, so these are pixel-space scene labels.
- AI visual silver is not human gold.
- Observed walkable polygons use pedestrian passage evidence; candidate goals are boundary priors, not future endpoints.
- AerialMPT supports short verified horizons here, not pedestrian t+50/t+100.
