# Stage42-IG T50 Gain/Harm Row Bootstrap

- source: `fresh_stage42_ig_t50_gain_harm_row_bootstrap`
- generated_at_utc: `2026-05-27T22:39:56.247935+00:00`
- input_hash: `203cf785b99b05183b7df91e697aa8eba633ceb68ca3b41b3a4e7430b0e7f67e`
- gate: `15 / 15`
- verdict: `stage42_ig_row_bootstrap_validates_selected_seed_with_multiseed_blocker`

## Purpose

Stage42-IF showed that Stage42-P has positive mean t+50 ADE but a negative 3-seed CI lower bound. This run recomputes row-level selected/fallback/oracle ADE/FDE from the cached Stage42-P checkpoints and performs bootstrap confidence intervals for the validation-selected seed.

## Claim Boundary

- dataset-local/raw-frame 2.5D only
- no true 3D claim
- no metric or seconds-level claim
- no Stage5C execution
- no SMC
- future waypoints are evaluation labels only, not inference inputs

## Validation-Selected Seed Bootstrap

| metric | value |
| --- | ---: |
| validation-selected seed | 151 |
| test rows | 55528 |
| t50 rows | 13689 |
| bootstrap n | 2000 |
| selected ADE t50 improvement | 0.028352 |
| selected ADE t50 CI low | 0.023371 |
| selected ADE t50 CI high | 0.033445 |
| selected FDE t50 improvement | 0.067566 |
| selected FDE t50 CI low | 0.060976 |
| selected ADE hard/failure improvement | 0.054677 |
| selected ADE easy degradation | 0.007574 |
| selected t50 two-action oracle headroom ADE | 0.096730 |
| multiseed ADE t50 CI low from Stage42-P | -0.017931 |

## Per-Seed Row Replay Metrics

| seed | ADE all | ADE t50 | ADE t50 CI low | FDE t50 | FDE t50 CI low | ADE hard | ADE easy degr | switch |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 149 | 0.044942 | 0.006432 | 0.001595 | 0.056683 | 0.051023 | 0.045274 | 0.013312 | 0.139911 |
| 151 | 0.051532 | 0.028352 | 0.023371 | 0.067566 | 0.060976 | 0.054677 | 0.007574 | 0.137138 |
| 157 | 0.058137 | -0.014996 | -0.019823 | 0.048042 | 0.043022 | 0.059817 | 0.004854 | 0.141280 |

## Interpretation

- Row-level bootstrap validates the validation-selected Stage42-P seed as a positive t+50 candidate.
- It does not erase the cross-seed instability found by Stage42-IF; the multiseed ADE t+50 CI lower bound remains negative.
- Therefore the paper-safe claim is: validation-selected row-level t+50 evidence is positive, while seed-stable ADE t+50 remains an open training-stability gap.
- The next research action is additional seeds or a more stable validation-selected policy family with row-error export enabled by default.
