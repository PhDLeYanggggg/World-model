# Source-Internal Validation Checkpoint Control

## Material Passport

fresh_run: 72 true Torch optimization runs, 144,000 updates including the 100-update pilot, 216 frozen causal directional decisions and full readout replay. cached_verified: upstream data and forecasters. Independent calibration/confirmation are not_run. No deployment change.

Primary validation-selected minus final signed-score MSE: -0.7068148302861347; nominal 95% locality CI [-0.9709775840747329, -0.45676933590876717]. Lower is better.
Matched ADE improvement over final: -0.5395324425995475%; nominal 95% locality CI [-0.7386193403569604, -0.34151630045464537].
Selected checkpoint steps: {'0': 21, '200': 20, '400': 4, '600': 2, '1000': 2, '1200': 6, '1400': 3, '1600': 5, '1800': 2, '2000': 7}. A step 0 selection is a training-prior decoder, not a learned improvement.
Seed matched-ADE gains: {'17': -0.6604248869698661, '29': -0.4919396169806791, '43': -0.4662328238480971}.

| Policy | All gain vs floor (%) | Easy gain vs floor (%) | Hard gain vs floor (%) | All-risk violations /216 | Easy-risk violations /216 | Undefined all/easy risk | Worst easy degradation (%) | Unknown interventions |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| final | 1.62667 | 0.80413 | 2.24408 | 197 | 198 | 0/0 | 28.56254 | 21672 |
| validation | 0.56809 | 2.49200 | 0.61540 | 122 | 149 | 21/21 | 21.05476 | 13500 |
| final_matched | 1.04453 | 1.11182 | 1.39969 | 177 | 177 | 21/21 | 25.03774 | 15084 |
| validation_matched | 0.52640 | 2.03985 | 0.59105 | 126 | 148 | 21/21 | 21.53520 | 11887 |

All summaries average within locality before across the 12 localities. The 216 role/seed views are dependent and development-exposed. The 3,000-draw bootstrap is nominal; it does not account for the complete adaptive research history. Unknown selected outcomes remain unknown. Undefined risk denominators do not pass a safety gate. Positive harm uses the unchanged 2% selected-reference budget, not net average ADE gain. Count matching need not match chosen sets.

## Reproduction

The first complete 2,000-update fit replays exactly excluding elapsed time; this does not retrain all 72 fits. All 216 action hashes are recreated and matched before each outcome readout. The complete readout also replays exactly. Checkpoints, normalization, sampler RNG, optimizer state, validation curves and both final/selected models remain local and excluded from Git. One disk-reserve interruption after the first fit was recovered from that checkpoint only after free space passed the original reserve again; no source or update was dropped.
Independent numerical checks: {'source_preprocessing': 72, 'validation_scores': 144, 'views': 216, 'metric_values': 27000, 'quality_scores': 432, 'query_count_checks': 747900, 'known_occurrences': 5614596, 'unknown_occurrences': 126846, 'unique_row_ids': 318969}.

## Limits

Recording-held source validation is a model-selection mechanism, not independent risk calibration. Changing the optimization/validation split means old full-source heads are not a matched causal comparison; only final versus selected within these runs is the primary contrast. No threshold search on transfer localities, independent-role opening or policy promotion. Image-local detector-silver, obs8/pred12 stride12 raw-frame only. No metric, seconds, human-gold, true3D, foundation, Stage5C or SMC claim.
