# Temporal Context Information Screen

## Material Passport
Fresh fitting-only ridge probes; cached-verified neural nuisance models. No neural retraining.
No outer-source scoring or independent evaluation. Detector pixels, obs8/pred12 annotation steps.

## Results

| Pair / contrast | Metric | Positive / negative / overlap / missing intervals | Point range |
|---|---|---|---|
| full / history_neighbors_vs_score_only | primary | 1 / 1 / 4 / 0 | [-2.2493886175644606, 4.525490396873495] |
| full / history_neighbors_vs_score_only | top10_pp | 0 / 1 / 5 / 0 | [-2.91993421657881, 3.129750332937419] |
| full / history_neighbors_vs_score_only | coverage_log | 2 / 0 / 4 / 0 | [0.026171265530018785, 0.2767618086716295] |
| full / history_neighbors_vs_score_only | all_harm | 0 / 0 / 6 / 0 | [0.0, 0.0] |
| full / history_neighbors_vs_old_summary | primary | 0 / 2 / 4 / 0 | [-2.0419126939662844, 0.16987863008537643] |
| full / history_neighbors_vs_old_summary | top10_pp | 0 / 0 / 6 / 0 | [-1.8824954335754887, -0.054276000228735466] |
| full / history_neighbors_vs_old_summary | coverage_log | 2 / 0 / 4 / 0 | [-0.03943149026222189, 0.08287858646197627] |
| full / history_neighbors_vs_old_summary | all_harm | 0 / 0 / 6 / 0 | [0.0, 0.0] |
| full / ordered_history_vs_score_only | primary | 1 / 1 / 4 / 0 | [-2.4082850839709233, 0.725327255992334] |
| full / ordered_history_vs_score_only | top10_pp | 0 / 0 / 6 / 0 | [-1.818897814380252, 2.4768974537936383] |
| full / ordered_history_vs_score_only | coverage_log | 3 / 0 / 3 / 0 | [0.012752828574591313, 0.19580914625530035] |
| full / ordered_history_vs_score_only | all_harm | 0 / 0 / 6 / 0 | [0.0, 0.0] |
| full / history_neighbors_vs_ordered_history | primary | 1 / 3 / 2 / 0 | [-1.2279792236396767, 3.9424876339784065] |
| full / history_neighbors_vs_ordered_history | top10_pp | 1 / 3 / 2 / 0 | [-1.287068774515006, 3.4193181683323046] |
| full / history_neighbors_vs_ordered_history | coverage_log | 1 / 1 / 4 / 0 | [-0.05401151230008205, 0.24370634296153873] |
| full / history_neighbors_vs_ordered_history | all_harm | 0 / 0 / 6 / 0 | [0.0, 0.0] |
| motion_only / history_neighbors_vs_score_only | primary | 3 / 0 / 3 / 0 | [-113.23934327490511, 3.7044745342048895] |
| motion_only / history_neighbors_vs_score_only | top10_pp | 2 / 0 / 2 / 2 | [-0.496786003648788, 18.134365850065254] |
| motion_only / history_neighbors_vs_score_only | coverage_log | 4 / 0 / 0 / 2 | [0.14567971791770012, 0.4501744773599382] |
| motion_only / history_neighbors_vs_score_only | all_harm | 0 / 0 / 6 / 0 | [0.0, 0.0] |
| motion_only / history_neighbors_vs_old_summary | primary | 1 / 5 / 0 / 0 | [-108.25062765168519, 0.47584810734391725] |
| motion_only / history_neighbors_vs_old_summary | top10_pp | 0 / 1 / 3 / 2 | [-3.2991548165351836, -1.0395221030437976] |
| motion_only / history_neighbors_vs_old_summary | coverage_log | 1 / 0 / 3 / 2 | [-0.06366044172453755, 0.038410158825301154] |
| motion_only / history_neighbors_vs_old_summary | all_harm | 0 / 0 / 6 / 0 | [0.0, 0.0] |
| motion_only / ordered_history_vs_score_only | primary | 2 / 1 / 3 / 0 | [-69.89221569504959, 7.273322852658677] |
| motion_only / ordered_history_vs_score_only | top10_pp | 3 / 0 / 1 / 2 | [0.9136071031754981, 19.69764925751839] |
| motion_only / ordered_history_vs_score_only | coverage_log | 3 / 0 / 1 / 2 | [0.06575007591338457, 0.48548897798961876] |
| motion_only / ordered_history_vs_score_only | all_harm | 0 / 0 / 6 / 0 | [0.0, 0.0] |
| motion_only / history_neighbors_vs_ordered_history | primary | 2 / 3 / 1 / 0 | [-11.441060691701583, 3.7299542500753815] |
| motion_only / history_neighbors_vs_ordered_history | top10_pp | 0 / 2 / 2 / 2 | [-3.9937450807728645, -1.4103931068242859] |
| motion_only / history_neighbors_vs_ordered_history | coverage_log | 1 / 1 / 2 / 2 | [-0.08680472742975147, 0.31611161170522895] |
| motion_only / history_neighbors_vs_ordered_history | all_harm | 0 / 0 / 6 / 0 | [0.0, 0.0] |

Primary is expected easy-harm MSE gain, not trajectory improvement or easy degradation.
Three outer-context differences are averaged within locality and seed, then three seeds within locality.
3000 paired resamples of four localities. Six assignments overlap; intervals are exploratory, not multiplicity adjusted.
Unknown or zero-denominator comparisons are retained as not_estimable, not dropped.

## Support

| Quantity | Minimum / median / maximum across dependent scoring views |
|---|---|
| supported_rows | {'min': 60, 'median': 2251.5, 'max': 73315} |
| easy_harm_rows | {'min': 0, 'median': 47.0, 'max': 5652} |
| recordings | {'min': 2, 'median': 4.0, 'max': 62} |
| event_recordings | {'min': 0, 'median': 4.0, 'max': 62} |
| tracks | {'min': 22, 'median': 767.5, 'max': 23539} |
| event_tracks | {'min': 0, 'median': 36.0, 'max': 3428} |
| agent_queries | {'min': 60, 'median': 2251.5, 'max': 73315} |
| event_agent_queries | {'min': 0, 'median': 47.0, 'max': 5652} |
| event_scene_queries | {'min': 0, 'median': 44.0, 'max': 4178} |
| event_nonoverlap_recording_queries | {'min': 0, 'median': 36.0, 'max': 2283} |
| event_track_mass_effective_count | {'min': 0.0, 'median': 12.237071563838054, 'max': 939.0641300583223} |
| largest_event_track_mass_share | {'min': 0.005068598730797084, 'median': 0.16779276629523152, 'max': 1.0} |

Counts above are not summed across overlapping cuts, seeds, producer assignments or outer contexts.
Non-overlap intervals and harm-mass effective counts are descriptive, not statistical sample-size estimates.
Distinct physical cohort: {"eu-locality-007": {"unique_recordings": 7, "unique_scene_queries": 1352, "unique_supported_agent_queries": 3893, "unique_tracks": 634}, "eu-locality-008": {"unique_recordings": 55, "unique_scene_queries": 12832, "unique_supported_agent_queries": 124291, "unique_tracks": 28787}, "eu-locality-020": {"unique_recordings": 4, "unique_scene_queries": 354, "unique_supported_agent_queries": 447, "unique_tracks": 74}, "eu-locality-048": {"unique_recordings": 4, "unique_scene_queries": 1010, "unique_supported_agent_queries": 5926, "unique_tracks": 1733}, "eu-locality-067": {"unique_recordings": 7, "unique_scene_queries": 1441, "unique_supported_agent_queries": 4905, "unique_tracks": 1177}, "eu-locality-074": {"unique_recordings": 62, "unique_scene_queries": 13650, "unique_supported_agent_queries": 78099, "unique_tracks": 14554}, "eu-locality-082": {"unique_recordings": 3, "unique_scene_queries": 492, "unique_supported_agent_queries": 782, "unique_tracks": 191}, "eu-locality-110": {"unique_recordings": 4, "unique_scene_queries": 902, "unique_supported_agent_queries": 7147, "unique_tracks": 1080}, "eu-locality-112": {"unique_recordings": 4, "unique_scene_queries": 892, "unique_supported_agent_queries": 4060, "unique_tracks": 1103}, "eu-locality-119": {"unique_recordings": 4, "unique_scene_queries": 893, "unique_supported_agent_queries": 4230, "unique_tracks": 1220}, "eu-locality-124": {"unique_recordings": 4, "unique_scene_queries": 597, "unique_supported_agent_queries": 2453, "unique_tracks": 507}, "eu-locality-126": {"unique_recordings": 4, "unique_scene_queries": 974, "unique_supported_agent_queries": 4576, "unique_tracks": 996}}

## Gates

- fitting_only_screen_complete: true
- exclusion_checked: true
- primary_screen: false
- protection_screen: false
- outer_context_study_justified: false
- independent_confirmation: false
- deployment_changed: false
- new_neural_training: false
- stage5c_executed: false
- smc_enabled: false

Cap diagnostics quantify the loss no easy-only correction can avoid while retaining the frozen all-harm cap.
They use evaluation labels only for analysis; never for inputs or intervention.
No metric/seconds, physical safety, human-gold, true3D or foundation claims. Stage5C/SMC remain off.
