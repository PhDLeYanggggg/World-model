# Matched Full-Source Native-Loss Forecasting

Fresh real Torch fits and held-source inference; source inputs hash-verified.
Four previously explored SDD auxiliary training scenes, not independent confirmation.
Full past-eligible population; unknown labels stay unknown. Pixel/raw-frame only.

| Objective | Seed | Native ADE gain vs CV (%) | Worst scene (%) | Complete sensitivity (%) |
| --- | ---: | ---: | ---: | ---: |
| past_normalized | 17 | 2.214390 | 1.293440 | 2.223424 |
| past_normalized | 29 | 2.139590 | 1.282464 | 2.138474 |
| past_normalized | 43 | 2.195775 | 1.348188 | 2.201968 |

past_normalized: mean-seed gain 2.183252%; conditional scene interval [1.586243220565947, 2.7515771126979756].

| native_coordinate | 17 | 7.725868 | 4.486444 | 8.000439 |
| native_coordinate | 29 | 7.874935 | 5.869037 | 8.198196 |
| native_coordinate | 43 | 7.298440 | 4.850209 | 7.577421 |

native_coordinate: mean-seed gain 7.633081%; conditional scene interval [5.96322487698451, 9.302157577621328].

Native versus matched old-loss model: 5.558312%.

All seeds, per-scene errors, tails, old metric and named slices are retained in analysis.json.
No best seed/checkpoint/threshold selected. Exactly static history is motion-bounded to CV;
missed static starts remain evaluated. Zero-CV easy percentage is undefined, not a full safety pass.
This is a loss contrast using an existing history/neighbor Transformer, not a new multimodal architecture.
No independent calibration or deployment; Stage5C/SMC disabled.
