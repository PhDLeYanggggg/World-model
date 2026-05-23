# Stage 21 User-Provided Data Intake

- No model training was run.
- Latent generative Stage 5C remains blocked.
- SMC remains blocked.
- Raw external data stays under `external_data/` and is ignored by git.

## OpenTraj

- exists: `True`
- toolkit exists: `True`
- dataset dirs: `28`
- SDD annotation files in OpenTraj tree: `60`
- ETH/UCY txt files in OpenTraj tree: `33`

## Kaggle SDD Archive

- archive exists: `True`
- archive files: `180`
- annotation files: `60`
- reference images: `60`
- videos: `60`

## Parsed SDD Summary

- scenes: `8`
- videos: `60`
- tracks: `10300`
- annotation rows: `10616256`
- max track length raw frames: `12272`
- raw-frame t+50 samples: `10101593`
- raw-frame t+100 samples: `9589470`
- coordinate status: `pixel-space; no homography/scale verified`
- effective seconds: not claimed until fps/video audit.

## License / Access

- Kaggle mirror reports CC0, but official Stanford SDD source/licensing is non-commercial/custom. The stricter/original-source status is recorded.
- Raw data must not be committed or redistributed.

## Next Step

1. Build Stage21 full SDD world-state rows from annotations.
2. Create train/val/test scene split and train-only candidate goal dictionaries.
3. Run full horizon/no-leakage audit before any deterministic retraining.
