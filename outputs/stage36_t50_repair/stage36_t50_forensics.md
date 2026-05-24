# Stage36 t+50 Forensics

- source: `fresh_run`; Stage35 split/labels are `cached_verified`.
- t+50 rows: `16263`
- t+50 oracle headroom: `0.22982786182653314`
- t+50 distribution: `{'easy': 6064, 'hard': 16263, 'failure': 987, 'hard_or_failure': 16263}`
- Stage35 t+50 switch rate: `0.0`
- Stage35 t+50 improvement: `0.0`
- predicted gain distribution: `{'mean': 0.00010847927884393488, 'p50': 0.0, 'p90': 0.0005168095231056214, 'positive_fraction': 0.19467502920740332}`
- fallback reasons: `{'fallback_easy_guard': 7023, 'fallback_no_predicted_gain': 9240}`
- baseline table: `{'constant_position': {'mean_fde': 1.2695345260932, 'mean_relative_fde': 0.32483425795139625}, 'constant_velocity_causal_fd': {'mean_fde': 0.689523068153967, 'mean_relative_fde': 0.18456547101745083}, 'damped_velocity': {'mean_fde': 0.7879498383445734, 'mean_relative_fde': 0.21634815135304533}, 'constant_acceleration_causal': {'mean_fde': 0.689523068153967, 'mean_relative_fde': 0.18456547101745083}, 'constant_turn_rate_velocity': {'mean_fde': 0.689523068153967, 'mean_relative_fde': 0.18456547101745083}, 'scene_clamped_baseline': {'mean_fde': 0.6854909319768842, 'mean_relative_fde': 0.18367537101761303}, 'goal_directed_baseline': {'mean_fde': 0.689523068153967, 'mean_relative_fde': 0.18456547101745083}}`
- all-test objective dilution: `{'t50_rows': 16263, 'all_rows': 66303, 't50_fraction': 0.24528301886792453, 'stage35_all_improvement': 0.12131890857784355, 'stage35_t50_improvement': 0.0, 'diagnosis': 'Stage35 validation objective captured short-horizon/t10 gains; t50 switch rate is effectively zero.'}`
- track length audit: `{'median': 20.0, 'p10': 20.0, 'p90': 20.0, 'note': 'track_length is audit metadata and is not used as an inference feature because full-track length may include future availability.'}`
