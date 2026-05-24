# Stage37 Conformal Safety Report

- source: `fresh_run`
- safe to deploy t50 selector: `True`
- final policy: `neighbor_history_t50_selector`
- final metrics: `{'rows': 66303, 'all_improvement': 0.1348254070727205, 't10_improvement': 0.30620476404756647, 't25_improvement': 0.0, 't50_improvement': 0.08457292542209705, 't100_improvement': 0.0, 'hard_failure_improvement': 0.1554340386904196, 'easy_degradation': 0.0004114683717719725, 'selector_regret': 0.493665392606631, 'harm_over_fallback': -0.14194807778019042, 'switch_rate': 0.0990452920682322, 'mean_confidence': 0.04121534153819084, 'stage35_non_t50_plus_stage37_t50': True}`
- The rule prioritizes easy degradation <=2% and harm_over_fallback <=0 before t50 lift.
