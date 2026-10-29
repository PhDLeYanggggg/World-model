# Completion-Aware Cost Supervision: Paired Training Results

## Material Passport

Fresh: 72 real Torch cost-head fits, 144,000 primary updates, first full refit,
144 fitting readout groups and separate exact replay. Cached_verified: source
packets, trajectory producers, physical normalizers and static references.
Not_run: new trajectory dynamics, held/independent evaluation or deployment.

Training PID 25627, exit 0: 296.702334s; peak RSS 10,179,477,504 B.
Readout PID 27079: 185.262481s; separate replay PID 27763: 188.511143s.
Native arm64 CPU4, workers0; no NumPy training fallback or resource probing.

## Actual Training Intervention

Known-only masks unknown-row loss. Completion mode uses explicit lower/upper
cost endpoints, not imputed observations. Original evaluation NaNs are unchanged.
Both modes share initial weights, normalization, source-balanced causal draws,
2,000 steps, architecture and optimizer. The new sampler differs from the older
known-only sampler, so both paired controls are freshly trained.

| Mode | Heads | Unknown draws | Unknown objective visits | Known objective visits |
|---|---:|---:|---:|---:|
| known_only | 36 | 348388 | 0 | 8867612 |
| completion_bounds | 36 | 348388 | 348388 | 8867612 |

These are repeated visits, not independent samples. Extra work: 2,000 full
refit updates and 180 pilot updates. The first complete model/Adam/RNG/trace
refit is exact; checkpoint bytes containing elapsed time need not match.

## Full Policy Results

| Mode | Policy | Known ADE gain vs floor (%), fitting 95% CI | Required risk support | Undefined easy | Known all/easy violations | Selected (unknown) |
|---|---|---|---:|---:|---:|---:|
| known_only | physical_only | +0.578053 [+0.227296, +1.033762] | 90/444 | 55 | 75/206 | 73184 (851) |
| known_only | physical_neural | +0.580799 [+0.270073, +0.978949] | 77/444 | 62 | 85/234 | 80414 (932) |
| known_only | neural_only | +0.579900 [+0.261588, +0.972832] | 126/444 | 95 | 117/113 | 23993 (314) |
| known_only | physical_only_static_parent | +0.671048 [+0.296152, +1.111184] | 73/444 | 50 | 85/229 | 77771 (901) |
| known_only | physical_neural_static_parent | +0.678064 [+0.320378, +1.094813] | 77/444 | 49 | 92/230 | 70561 (853) |
| known_only | frozen_origin_binary | +7.934961 [+5.437828, +10.345468] | 16/444 | 73 | 331/164 | 62902 (1567) |
| known_only | floor | +0.000000 [+0.000000, +0.000000] | 0/444 | 444 | 0/0 | 0 (0) |
| completion_bounds | physical_only | -20.047912 [-54.061613, -0.667975] | 43/444 | 8 | 281/362 | 235283 (3400) |
| completion_bounds | physical_neural | -18.512092 [-50.358572, -0.239583] | 26/444 | 8 | 311/375 | 237281 (3449) |
| completion_bounds | neural_only | +0.579900 [+0.261588, +0.972832] | 126/444 | 95 | 117/113 | 23993 (314) |
| completion_bounds | physical_only_static_parent | +0.671048 [+0.296152, +1.111184] | 73/444 | 50 | 85/229 | 77771 (901) |
| completion_bounds | physical_neural_static_parent | +0.678064 [+0.320378, +1.094813] | 77/444 | 49 | 92/230 | 70561 (853) |
| completion_bounds | frozen_origin_binary | +7.934961 [+5.437828, +10.345468] | 16/444 | 73 | 331/164 | 62902 (1567) |
| completion_bounds | floor | +0.000000 [+0.000000, +0.000000] | 0/444 | 444 | 0/0 | 0 (0) |

Cached static policies and the neural-only head are secondary references,
not new models. Empty selected risk remains undefined; the floor is not credited
as a passed selected-risk estimate. No required recording is dropped.

## Completion Minus Matched Known-Only Training

| Bank | Full known change (pp), fitting CI | Full completion-lower (pp), fitting CI | Matched completion-lower (pp), fitting CI |
|---|---|---|---|
| physical_only | -20.625965 [-54.598811, -1.442137] | -21.340839 [-55.387016, -1.798143] | -0.569496 [-1.602558, +0.173903] |
| physical_neural | -19.092891 [-50.819548, -1.021775] | -19.857353 [-51.911894, -1.420127] | -0.351109 [-1.188380, +0.348592] |

Count matching concerns utility only. Completion utility uses fixed known
floor-error mass, not an identified complete-population error denominator.

## Neural Bank Increment

| Mode | Mixed minus physical known (pp), fitting CI | Completion-lower (pp), fitting CI |
|---|---|---|
| known_only | +0.002746 [-0.066949, +0.064489] | -0.033985 [-0.095423, +0.013719] |
| completion_bounds | +1.535820 [+0.223005, +3.881457] | +1.266730 [-0.092145, +3.645457] |

## Endpoint-Proxy Fit Diagnostics

Pooled normalized squared error over real action occurrences; not training loss,
a source-macro estimate or a conditional-risk certificate. Unknown targets are
interval endpoints, not observed future errors. Undefined group counts remain.

| Mode | Bank | Label support | Real actions | Undefined groups | B_lower / H_upper / R_lower / ER_lower / EH_upper MSE |
|---|---|---|---:|---:|---|
| known_only | physical_only | all | 4474875 | 0 | 0.269061, 1.65931, 1.86567, 0.00292431, 1.69425 |
| known_only | physical_only | known | 4380675 | 0 | 0.268674, 0.558567, 1.84446, 0.00296777, 0.0259229 |
| known_only | physical_only | unknown | 94200 | 0 | 0.287037, 52.8483, 2.85208, 0.000903071, 79.2782 |
| known_only | physical_neural | all | 5369850 | 0 | 0.266044, 1.58714, 1.84927, 0.00289618, 1.59606 |
| known_only | physical_neural | known | 5256810 | 0 | 0.265216, 0.506626, 1.82582, 0.00293979, 0.023455 |
| known_only | physical_neural | unknown | 113040 | 0 | 0.304555, 51.8355, 2.93978, 0.00086781, 74.7285 |
| completion_bounds | physical_only | all | 4474875 | 0 | 0.316543, 1.92622, 2.05564, 0.00303655, 1.49303 |
| completion_bounds | physical_only | known | 4380675 | 0 | 0.32012, 1.25046, 2.07186, 0.00308, 0.250654 |
| completion_bounds | physical_only | unknown | 94200 | 0 | 0.150211, 33.3518, 1.30173, 0.00101611, 59.2684 |
| completion_bounds | physical_neural | all | 5369850 | 0 | 0.310461, 1.79873, 2.03922, 0.00302771, 1.41545 |
| completion_bounds | physical_neural | known | 5256810 | 0 | 0.313669, 1.12488, 2.05543, 0.00307203, 0.226372 |
| completion_bounds | physical_neural | unknown | 113040 | 0 | 0.161276, 33.1353, 1.28553, 0.00096638, 56.7123 |

## Per-Locality Reference Sensitivity

Primary relative metrics retain every locality. The native values below average
repeated context/head-seed view means in pixels; they are a denominator diagnostic,
not a substitute pooled endpoint, new split or independent observation count.

| Locality | Known-only mixed gain (%) | Completion mixed gain (%) | Known-only floor / policy ADE pixels | Completion floor / policy ADE pixels |
|---|---:|---:|---|---|
| eu-locality-007 | +0.739506 | +1.106337 | 13.1711 / 13.0733 | 13.1711 / 13.0246 |
| eu-locality-008 | +0.737835 | +0.259762 | 17.8036 / 17.6722 | 17.8036 / 17.7573 |
| eu-locality-020 | +0.866024 | +0.291330 | 9.39795 / 9.31656 | 9.39795 / 9.37057 |
| eu-locality-048 | +0.119262 | +0.027278 | 21.5463 / 21.5206 | 21.5463 / 21.5405 |
| eu-locality-067 | +0.404059 | +0.168767 | 25.6607 / 25.557 | 25.6607 / 25.6174 |
| eu-locality-074 | +0.449119 | -23.115443 | 18.9162 / 18.8311 | 18.9162 / 23.2874 |
| eu-locality-082 | +0.000000 | -179.159432 | 21.8383 / 21.8383 | 21.8383 / 60.9638 |
| eu-locality-110 | +2.408308 | +5.790355 | 42.3161 / 41.2969 | 42.3161 / 39.8653 |
| eu-locality-112 | +0.079986 | -22.118566 | 22.1323 / 22.1146 | 22.1323 / 27.0275 |
| eu-locality-119 | +0.153913 | +0.468641 | 31.9009 / 31.8517 | 31.9009 / 31.7521 |
| eu-locality-124 | +0.042109 | +0.009585 | 30.8618 / 30.8488 | 30.8618 / 30.8589 |
| eu-locality-126 | +0.969471 | -5.873715 | 18.9272 / 18.744 | 18.9272 / 20.0427 |

## Limits And Decision

All intervals use 3,000 fitting-locality resamples after averaging repeated
context/head-seed views. They are fitting resubstitution, not independent
generalization, whole-pipeline OOF or three forecasting pipelines. Known FDE
aggregation with any required undefined denominator remains not_estimable.
Supplementary pooled hard/easy values do not replace the full recording gate.

Valid interval endpoints do not make a finite fitted cost head calibrated after
selection. This comparison fails the fitting advancement gate. No thresholds,
seeds or checkpoints are selected to hide that result. See failure_analysis.md
and the separate same-region diagnosis; neither changes the original verdict.

SOURCE is oblique detector-track midpoint pixels, obs8/pred12/stride12 raw
frames, not SDD human gold, metric/seconds, true 3D or foundation evidence.
Independent roles and DroneCrowd remain closed. No Stage5C or SMC.

## Gates

- full144fitting_groups: true
- full_mixed_recording_risk: false
- full_completion_minus_control_gain_CI_positive: false
- matched_completion_minus_control_gain_CI_positive: false
- mixed_minus_physical_completion_CI_positive: false
- fitting_advancement_gate: false
- independent_confirmation: false
- deployment_changed: false
- submission_ready: false
- stage5c_executed: false
- smc_enabled: false

[Loss curves](figures/loss_traces.svg), [policy comparison](figures/policy_comparison.svg),
[Chinese reproduction guide](reproduction_zh.md), [training traces](training_traces.json).
