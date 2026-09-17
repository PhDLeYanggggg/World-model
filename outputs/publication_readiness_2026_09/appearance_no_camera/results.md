# Past-RGB Retraining Without Camera Inputs

All six fixed-budget fits, no seed/threshold selection. Positive gain means better than CV.

| Held scene | Seed | Original guarded gain % | No-camera guarded gain % | No-camera unrestricted gain % | Switch rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| ETH | 17 | -14.35999 | -38.40763 | -37.08570 | 66.6667% |
| ETH | 29 | -8.91739 | -5.56005 | -4.96272 | 64.1975% |
| ETH | 43 | -21.65083 | -14.93887 | -14.79149 | 66.6667% |
| Hotel | 17 | -124.64757 | -46.15188 | -317.70849 | 15.8451% |
| Hotel | 29 | -198.93458 | -89.45431 | -439.02063 | 22.1831% |
| Hotel | 43 | -194.31460 | -123.22167 | -435.62121 | 28.1690% |

Easy percentage ratios undefined at zero CV floor; absolute harm/native errors in metrics.json.
Fit-only adaptive research, not independent confirmation, metric calibration or deployment.
