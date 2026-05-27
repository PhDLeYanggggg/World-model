# Stage42-IH T50 Gain/Harm Seed Expansion

- source: `fresh_stage42_ih_t50_gain_harm_seed_expansion`
- generated_at_utc: `2026-05-27T23:07:34.159261+00:00`
- input_hash: `45ce7b33fa9cde93291bed191cc802f187f78f12355c544f1d788424d901df6a`
- gate: `15 / 16`
- verdict: `stage42_ih_t50_seed_expansion_mean_positive_ci_blocker_remains`

## Purpose

Stage42-IF found seed-level ADE t+50 instability, while Stage42-IG validated the validation-selected seed with row bootstrap. Stage42-IH adds more t+50 gain/harm selector seeds to test whether the mean t+50 gain becomes seed-stable.

## Claim Boundary

- dataset-local/raw-frame 2.5D only
- no true 3D, metric, seconds-level, or foundation claim
- no Stage5C execution
- no SMC
- future waypoints are train/val labels and final evaluation labels only, not inference inputs

## Seed Expansion Summary

| metric | original 3 seeds | expanded seeds |
| --- | ---: | ---: |
| seed count | 3 | 6 |
| ADE all mean [CI] | 0.051537 [0.044071, 0.059003] | 0.051911 [0.046703, 0.057118] |
| ADE t50 mean [CI] | 0.006596 [-0.017931, 0.031123] | 0.006727 [-0.008183, 0.021636] |
| ADE t100 raw diag mean [CI] | 0.059254 [0.021213, 0.097295] | 0.060537 [0.036196, 0.084878] |
| ADE hard/failure mean [CI] | 0.053256 [0.044911, 0.061602] | 0.053456 [0.047993, 0.058919] |
| ADE easy degradation mean [CI] | 0.008580 [0.003694, 0.013466] | 0.011112 [0.005632, 0.016593] |
| FDE t50 mean [CI] | 0.057431 [0.046360, 0.068502] | 0.059987 [0.054343, 0.065631] |

## Validation-Selected Combined Seed

- selected: `True`
- seed: `151`
- base_seed: `113`
- validation_score: `1.001763`
- test ADE all/t50/hard/easy: `0.051532` / `0.028352` / `0.054677` / `0.007574`
- test FDE t50: `0.067566`

## Per-Seed Test Metrics

| seed | source | base_seed | val_score | ADE all | ADE t50 | ADE t100 raw | ADE hard | ADE easy degr | FDE t50 | switch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 149 | `fresh_run` | 109 | 0.961670 | 0.044942 | 0.006432 | 0.020561 | 0.045274 | 0.013312 | 0.056683 | 0.139911 |
| 151 | `fresh_run` | 113 | 1.001763 | 0.051532 | 0.028352 | 0.075921 | 0.054677 | 0.007574 | 0.067566 | 0.137138 |
| 157 | `fresh_run` | 127 | 0.588214 | 0.058137 | -0.014996 | 0.081281 | 0.059817 | 0.004854 | 0.048042 | 0.141280 |
| 163 | `fresh_run` | 109 | 0.997965 | 0.044017 | -0.000458 | 0.022370 | 0.045103 | 0.023085 | 0.059696 | 0.145656 |
| 167 | `fresh_run` | 113 | 0.984475 | 0.053151 | 0.029437 | 0.084902 | 0.055259 | 0.005381 | 0.065694 | 0.137822 |
| 173 | `fresh_run` | 127 | 0.608386 | 0.059686 | -0.008404 | 0.078187 | 0.060606 | 0.012467 | 0.062240 | 0.142703 |

## Domain Instability

- original negative t50 slices: `3`
- expanded negative t50 slices: `6`

Worst expanded t+50 slices:

| seed | domain | rows | all | t50 | hard/failure | easy degradation | switch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 173 | `TrajNet` | 20087 | 0.114400 | -0.048201 | 0.115804 | 0.035342 | 0.222034 |
| 157 | `TrajNet` | 20087 | 0.112792 | -0.046966 | 0.116043 | 0.010668 | 0.227709 |
| 163 | `TrajNet` | 20087 | 0.071161 | -0.009917 | 0.071257 | 0.057336 | 0.230199 |
| 151 | `ETH_UCY` | 25901 | 0.009729 | -0.004499 | 0.009528 | 0.008378 | 0.133740 |
| 167 | `ETH_UCY` | 25901 | 0.014717 | -0.002602 | 0.014672 | 0.000000 | 0.126289 |

## Interpretation

- This stage expands selector random seeds while reusing the three cached Stage42-N base checkpoints; that reuse is deliberate and explicitly disclosed.
- If the combined ADE t+50 CI lower bound is positive, Stage42-P's t+50 selector has stronger seed-level evidence.
- If it remains negative, the open blocker is not row bootstrap but train-seed/domain instability, and the next step should alter the policy family or training objective.
