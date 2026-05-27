# Stage42-IF T50 Gain/Harm Stability Audit

- source: `fresh_stage42_if_t50_gain_harm_stability_audit`
- generated_at_utc: `2026-05-27T22:24:48.574802+00:00`
- input_hash: `203cf785b99b05183b7df91e697aa8eba633ceb68ca3b41b3a4e7430b0e7f67e`
- gate: `13 / 14`
- verdict: `stage42_if_t50_gain_harm_ci_blocker_identified`

## What This Audits

Stage42-P repaired the mean t+50 signal for the gain/harm selector. This audit checks whether that result is stable enough to use as a paper-level t+50 ADE claim.

## Current Facts

- This is still dataset-local/raw-frame 2.5D multi-agent world-state evidence.
- It is not true 3D, not metric, not seconds-level, and not a foundation world model.
- Stage5C latent generative execution remains disabled.
- SMC remains disabled.

## Seed-Level Summary

| metric | value |
| --- | ---: |
| seeds | 3 |
| ADE t50 mean | 0.006596 |
| ADE t50 CI low | -0.017931 |
| ADE t50 CI high | 0.031123 |
| FDE t50 mean | 0.057431 |
| FDE t50 CI low | 0.046360 |
| negative ADE t50 seeds | 1 |
| paper-stable ADE t50 claim supported | `False` |
| row-level bootstrap status | `not_run_blocked_by_missing_row_errors_in_stage42p_artifact` |

## Per-Seed Test ADE

| seed | base_seed | all | t50 | t100 raw diag | hard/failure | easy degradation | validation score |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 149 | 109 | 0.044942 | 0.006432 | 0.020561 | 0.045274 | 0.013312 | 0.961670 |
| 151 | 113 | 0.051532 | 0.028352 | 0.075921 | 0.054677 | 0.007574 | 1.001763 |
| 157 | 127 | 0.058137 | -0.014996 | 0.081281 | 0.059817 | 0.004854 | 0.588214 |

## Validation-Selected Seed

- selected_seed: `151`
- selection rule: validation-only t+50-weighted score; no test threshold tuning.
- selected test ADE all: `0.051532`
- selected test ADE t50: `0.028352`
- selected test ADE hard/failure: `0.054677`
- selected test easy degradation: `0.007574`

This is useful deployment-selection evidence, but it is not a substitute for stable multi-seed or row-bootstrap evidence.

## Worst Domain T50 Slices

| seed | domain | rows | all | t50 | hard/failure | easy degradation | switch |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 157 | `TrajNet` | 20087 | 0.112792 | -0.046966 | 0.116043 | 0.010668 | 0.227709 |
| 151 | `ETH_UCY` | 25901 | 0.009729 | -0.004499 | 0.009528 | 0.008378 | 0.133740 |
| 157 | `ETH_UCY` | 25901 | 0.048162 | -0.001826 | 0.049987 | 0.002243 | 0.126289 |
| 149 | `UCY` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 151 | `UCY` | 9540 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Diagnosis

- primary_blocker: `seed_level_ade_t50_instability`
- Stage42-P improves mean t+50 ADE but one seed is negative, so the seed-level CI lower bound is below zero.
- TrajNet t+50 is the main unstable domain slice: it is strongly positive for seed 151 but negative for seed 157.
- Existing Stage42-P artifact stores aggregate metrics, not per-row error vectors, so row bootstrap cannot be recomputed without rerunning/cache export.
- FDE t+50 is stable positive, which means endpoint-distance style evidence is stronger than ADE-over-horizon evidence.

## Next Action

- rerun t50 gain/harm selector with exported per-row selected/fallback/oracle error arrays and additional validation-selected seeds before promoting a paper-level ADE t50 claim
