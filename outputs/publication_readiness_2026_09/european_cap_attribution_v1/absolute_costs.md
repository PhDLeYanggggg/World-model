# Absolute Cost Companion

Post-screen descriptions of every fixed arm, not a new comparison or selection criterion.
Each min/median/max summarizes216 dependent views; medians are not paired effects or pooled error.
MSE is squared native pixel-derived expected-cost error, not trajectory ADE/FDE.

| Family / arm | Easy-harm MSE min / median / max | All-harm MSE min / median / max |
|---|---|---|
| full / score_only_nonnegative | {'min': 0.0015344669327263395, 'median': 0.047836547926637225, 'max': 2.0515015384144637} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / score_only_frozen_cap | {'min': 0.001509958866276074, 'median': 0.046862884782281906, 'max': 2.053056322969861} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / score_only_envelope | {'min': 0.0015344669327263395, 'median': 0.047835805528205516, 'max': 2.0515015384144637} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / score_only_envelope_coupled | {'min': 0.0015344669327263395, 'median': 0.047835805528205516, 'max': 2.0515015384144637} | {'min': 0.4097560582484889, 'median': 3.2123305068068237, 'max': 55.33003366884102} |
| full / old_summary_nonnegative | {'min': 0.001504866385038018, 'median': 0.045950494112179455, 'max': 2.045094829551997} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / old_summary_frozen_cap | {'min': 0.0014826966282298164, 'median': 0.045774071137788384, 'max': 2.048105664962825} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / old_summary_envelope | {'min': 0.001504866385038018, 'median': 0.04595027263588286, 'max': 2.045094829551997} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / old_summary_envelope_coupled | {'min': 0.001504866385038018, 'median': 0.04595027263588286, 'max': 2.045094829551997} | {'min': 0.40987412261064937, 'median': 3.212282681944181, 'max': 55.330268461605115} |
| full / ordered_history_nonnegative | {'min': 0.0015489046771829892, 'median': 0.047092847073836284, 'max': 2.0511671215762655} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / ordered_history_frozen_cap | {'min': 0.0015058555850509438, 'median': 0.04646425868515292, 'max': 2.0521010388046994} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / ordered_history_envelope | {'min': 0.0015489046771829892, 'median': 0.047092847073836284, 'max': 2.0511671215762655} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / ordered_history_envelope_coupled | {'min': 0.0015489046771829892, 'median': 0.047092847073836284, 'max': 2.0511671215762655} | {'min': 0.41003805511682584, 'median': 3.2123104788430004, 'max': 55.33020506350706} |
| full / history_neighbors_nonnegative | {'min': 0.0015629452103758444, 'median': 0.046378189365928885, 'max': 2.0483849141198323} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / history_neighbors_frozen_cap | {'min': 0.001514238115529214, 'median': 0.04605273597030712, 'max': 2.051874809585315} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / history_neighbors_envelope | {'min': 0.0015629452103758444, 'median': 0.04637646287914012, 'max': 2.0483849141198323} | {'min': 0.4097909129376051, 'median': 3.212583821436586, 'max': 55.331487042531926} |
| full / history_neighbors_envelope_coupled | {'min': 0.0015629452103758444, 'median': 0.04637646287914012, 'max': 2.0483849141198323} | {'min': 0.4100987133734617, 'median': 3.212365591159557, 'max': 55.330427155310424} |
| motion_only / score_only_nonnegative | {'min': 9.396765133593241e-07, 'median': 0.02953239723985853, 'max': 2.7731368582524114} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / score_only_frozen_cap | {'min': 9.030415776003912e-07, 'median': 0.026641168152210397, 'max': 2.1282178769499946} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / score_only_envelope | {'min': 9.396765133593241e-07, 'median': 0.02953239723985853, 'max': 2.7654834599612252} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / score_only_envelope_coupled | {'min': 9.396765133593241e-07, 'median': 0.02953239723985853, 'max': 2.7654834599612252} | {'min': 0.3160492337480426, 'median': 2.9039423051892204, 'max': 54.78353004748298} |
| motion_only / old_summary_nonnegative | {'min': 7.060469538908329e-07, 'median': 0.026630412302847975, 'max': 2.844875529217721} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / old_summary_frozen_cap | {'min': 6.871030736208608e-07, 'median': 0.026630250812429458, 'max': 2.1239261643853564} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / old_summary_envelope | {'min': 7.060469538908329e-07, 'median': 0.026630412302847975, 'max': 2.8342635965491394} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / old_summary_envelope_coupled | {'min': 7.060469538908329e-07, 'median': 0.026630412302847975, 'max': 2.8342635965491394} | {'min': 0.31603271628641777, 'median': 2.903810140480193, 'max': 54.7822391525932} |
| motion_only / ordered_history_nonnegative | {'min': 9.158604783365274e-06, 'median': 0.029677270066130322, 'max': 2.587458911497576} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / ordered_history_frozen_cap | {'min': 8.989521026427418e-06, 'median': 0.0265979493698836, 'max': 2.124297322839834} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / ordered_history_envelope | {'min': 9.158604783365274e-06, 'median': 0.029677270066130322, 'max': 2.5839184619894517} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / ordered_history_envelope_coupled | {'min': 9.158604783365274e-06, 'median': 0.029677270066130322, 'max': 2.5839184619894517} | {'min': 0.31603021131110853, 'median': 2.9037103618971987, 'max': 54.781005909093544} |
| motion_only / history_neighbors_nonnegative | {'min': 1.80904249935747e-05, 'median': 0.027991432100912813, 'max': 2.9723796797763153} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / history_neighbors_frozen_cap | {'min': 1.2199630110056581e-05, 'median': 0.026640732211865477, 'max': 2.125211174974706} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / history_neighbors_envelope | {'min': 1.80904249935747e-05, 'median': 0.027991432100912813, 'max': 2.956735470361861} | {'min': 0.31614690761713604, 'median': 2.89088972825226, 'max': 54.784970507714014} |
| motion_only / history_neighbors_envelope_coupled | {'min': 1.80904249935747e-05, 'median': 0.027991432100912813, 'max': 2.956735470361861} | {'min': 0.3160408362023172, 'median': 2.903599163290923, 'max': 54.780220097843426} |

Large relative percentages may reflect tiny denominators. No unfavorable arm is removed.
A pointwise label-derived ceiling is not evidence of conditional-mean bias.
The registered paired intervals, not these descriptive medians, determine the information screen.
No independent outcomes, refitting, trajectory deployment, metric/seconds, Stage5C or SMC.

## Post-Hoc View Counts

Counts describe overlapping views, not independent votes or a new gate.
A zero easy-harm label includes non-easy samples as well as non-harm events; it is not synonymous with an easy case.

| Family / arm | Dependent view counts |
|---|---|
| full / score_only | {'dependent_views': 216, 'lower_easy_MSE': 86, 'higher_easy_MSE': 130, 'unchanged_easy_MSE': 0, 'event_benefit_positive': 202, 'event_benefit_but_total_worse': 116, 'zero_event_contribution_nonpositive': True, 'coupled_all_harm_MSE_worse': 34} |
| full / old_summary | {'dependent_views': 216, 'lower_easy_MSE': 71, 'higher_easy_MSE': 145, 'unchanged_easy_MSE': 0, 'event_benefit_positive': 200, 'event_benefit_but_total_worse': 129, 'zero_event_contribution_nonpositive': True, 'coupled_all_harm_MSE_worse': 25} |
| full / ordered_history | {'dependent_views': 216, 'lower_easy_MSE': 69, 'higher_easy_MSE': 147, 'unchanged_easy_MSE': 0, 'event_benefit_positive': 202, 'event_benefit_but_total_worse': 133, 'zero_event_contribution_nonpositive': True, 'coupled_all_harm_MSE_worse': 38} |
| full / history_neighbors | {'dependent_views': 216, 'lower_easy_MSE': 62, 'higher_easy_MSE': 154, 'unchanged_easy_MSE': 0, 'event_benefit_positive': 203, 'event_benefit_but_total_worse': 141, 'zero_event_contribution_nonpositive': True, 'coupled_all_harm_MSE_worse': 37} |
| motion_only / score_only | {'dependent_views': 216, 'lower_easy_MSE': 55, 'higher_easy_MSE': 157, 'unchanged_easy_MSE': 4, 'event_benefit_positive': 128, 'event_benefit_but_total_worse': 73, 'zero_event_contribution_nonpositive': True, 'coupled_all_harm_MSE_worse': 32} |
| motion_only / old_summary | {'dependent_views': 216, 'lower_easy_MSE': 55, 'higher_easy_MSE': 160, 'unchanged_easy_MSE': 1, 'event_benefit_positive': 134, 'event_benefit_but_total_worse': 79, 'zero_event_contribution_nonpositive': True, 'coupled_all_harm_MSE_worse': 28} |
| motion_only / ordered_history | {'dependent_views': 216, 'lower_easy_MSE': 51, 'higher_easy_MSE': 164, 'unchanged_easy_MSE': 1, 'event_benefit_positive': 140, 'event_benefit_but_total_worse': 89, 'zero_event_contribution_nonpositive': True, 'coupled_all_harm_MSE_worse': 23} |
| motion_only / history_neighbors | {'dependent_views': 216, 'lower_easy_MSE': 47, 'higher_easy_MSE': 168, 'unchanged_easy_MSE': 1, 'event_benefit_positive': 134, 'event_benefit_but_total_worse': 87, 'zero_event_contribution_nonpositive': True, 'coupled_all_harm_MSE_worse': 25} |
