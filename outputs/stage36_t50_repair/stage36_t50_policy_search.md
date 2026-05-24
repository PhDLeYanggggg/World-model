# Stage36 t+50 Conservative Policy Search

- source: `fresh_run`
- t50 selector enabled: `False`
- t50 selector test metrics: `{'rows': 66303, 'all_improvement': -0.0018587637543769908, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': -0.01163892243691822, 't100_improvement': 0.0, 'hard_failure_improvement': -0.002144751142444834, 'easy_degradation': 0.0, 'selector_regret': 0.5586280635657679, 'harm_over_fallback': 0.0019569600990635046, 'switch_rate': 0.049047554409302745, 'mean_confidence': 0.009633413515985012}`
- final policy: `stage35_selective_transfer_with_t50_fallback`
- final test metrics: `{'rows': 66303, 'all_improvement': 0.12131890857784355, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.0, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1398494448930071, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.4289430569144169, 'harm_over_fallback': -0.12772804655228734, 'switch_rate': 0.04999773765892946, 'mean_confidence': 0.03720055893063545}`
- If the val-selected t50 selector cannot exceed the 3% t50 gate safely, Stage36 falls back on t+50 instead of forcing harmful switches.
