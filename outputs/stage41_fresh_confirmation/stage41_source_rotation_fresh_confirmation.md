# Stage41 Source-Rotation Fresh Confirmation

- source: `fresh_run`
- protocol status: `source_rotation_fresh_confirmation`
- deployment decision: `stage41_neural_fresh_confirmed_partial_not_full_replacement`
- fresh confirmation pass: `True`
- Stage37 any-core margin pass: `True`
- Stage37 all-core margin pass: `False`
- full replacement pass: `False`
- best name: `fresh_rotation_ensemble::relaxed_easy_budget`
- positive external domains: `2`
- max domain easy degradation: `0.0`
- t50 oracle ceiling: `0.07570014620278032`
- t50 oracle below Stage37: `True`
- best metrics: `{'rows': 55528, 'all_improvement': 0.20881762937561832, 't10_improvement': 0.0001793565242457218, 't25_improvement': 0.039457337027849704, 't50_improvement': 0.05448600669657733, 't100_improvement': 0.4572355026149352, 'hard_failure_improvement': 0.22538184888542845, 'easy_degradation': 0.0, 'harm_over_fallback': -0.14354694508120913, 'switch_rate': 0.12764731306728136, 'regret_to_oracle': 0.04414118101599986, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': 0.2538593034850489, 't50_improvement': 0.06735455924940559, 't100_improvement': 0.5385035854281408, 'hard_failure_improvement': 0.2691451906066811, 'easy_degradation': 0.0, 'switch_rate': 0.1775993204895564}, 'TrajNet': {'rows': 20087, 'all_improvement': 0.26417358132428503, 't50_improvement': 0.06567822414892932, 't100_improvement': 0.6061667533205937, 'hard_failure_improvement': 0.29057526998285066, 'easy_degradation': 0.0, 'switch_rate': 0.12386120376362822}, 'UCY': {'rows': 9540, 'all_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.0, 'easy_degradation': 0.0, 'switch_rate': 0.0}}, 'candidate_oracle': {'rows': 55528, 'all_improvement': 0.27302977107174786, 't10_improvement': 0.0011265454723250468, 't25_improvement': 0.05685015801169491, 't50_improvement': 0.07570014620278032, 't100_improvement': 0.5924008882347781, 'hard_failure_improvement': 0.29376902671450755, 'easy_degradation': 0.0, 'harm_over_fallback': -0.18768812609720897, 'switch_rate': 1.0, 'regret_to_oracle': 0.0, 'by_domain': {'ETH_UCY': {'rows': 25901, 'all_improvement': 0.26591154288040475, 't50_improvement': 0.0766776163497408, 't100_improvement': 0.5574296097976473, 'hard_failure_improvement': 0.28100750528307394, 'easy_degradation': 0.0, 'switch_rate': 1.0}, 'TrajNet': {'rows': 20087, 'all_improvement': 0.2724864124809556, 't50_improvement': 0.0735275437384546, 't100_improvement': 0.6184879138052229, 'hard_failure_improvement': 0.2985883338538917, 'easy_degradation': 0.0, 'switch_rate': 1.0}, 'UCY': {'rows': 9540, 'all_improvement': 0.2932943815841236, 't50_improvement': 0.0763795492531959, 't100_improvement': 0.6531018662747778, 'hard_failure_improvement': 0.32253747159218404, 'easy_degradation': 0.0, 'switch_rate': 1.0}}}, 'selected_candidate_distribution': {'0': 48440, '1': 1536, '2': 112, '3': 359, '4': 1, '6': 4204, '7': 77, '8': 799}, 't50_ci': {'low': 0.050008009630071015, 'mid': 0.054521789980955304, 'high': 0.05901264903516945, 'n': 13689}, 'hard_failure_ci': {'low': 0.21991911352206958, 'mid': 0.2254047278632852, 'high': 0.23080080070208137, 'n': 41741}}`
- caveat: `This is a source-rotation confirmation on existing external datasets, not a new dataset. UCY independent rotation is limited by duplicated zara03 files and only two unique UCY scenes. The fresh run confirms all/hard neural lift but does not fully replace Stage37 because t50 remains below Stage37 and the candidate-oracle t50 ceiling is also below Stage37 on this rotation.`

Strict claims:

- true 3D world model: `False`
- foundation world model: `False`
- metric/seconds-level claim: `False`
- Stage5C executed: `False`
- SMC enabled: `False`
