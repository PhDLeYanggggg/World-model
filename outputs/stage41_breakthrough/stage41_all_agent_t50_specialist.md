# Stage41 All-Agent t50 Specialist

- source: `fresh_run`
- best trial: `t50_hard_gain_harm`
- deployment: `diagnostic_keep_m3w_neural_v1_endpoint_candidate`
- metrics: `{'rows': 34777, 'all_improvement': 0.023127391643180673, 't10_improvement': 0.0, 't25_improvement': 0.0, 't50_improvement': 0.09375204966816386, 't100_improvement': 0.0, 'hard_failure_improvement': 0.02472403070303797, 'easy_degradation': 0.0, 'harm_over_fallback': -0.032593482585597675, 'switch_rate': 0.05095321620611323, 'regret_to_oracle': 0.4985898613700518, 'by_domain': {'ETH_UCY': {'rows': 21598, 'all_improvement': 0.013160082077096624, 't50_improvement': 0.05568545014429005, 't100_improvement': 0.0, 'hard_failure_improvement': 0.013942827929712687, 'easy_degradation': 0.0, 'switch_rate': 0.04694879155477359}, 'TrajNet': {'rows': 3639, 'all_improvement': 0.06486501302696968, 't50_improvement': 0.22484555930596373, 't100_improvement': 0.0, 'hard_failure_improvement': 0.07105692941524677, 'easy_degradation': 0.0, 'switch_rate': 0.07969222313822479}, 'UCY': {'rows': 9540, 'all_improvement': 0.05913005433415497, 't50_improvement': 0.2076505046017627, 't100_improvement': 0.0, 'hard_failure_improvement': 0.06572922198499065, 'easy_degradation': 0.0, 'switch_rate': 0.04905660377358491}}, 'selected_candidate_distribution': {0: 33005, -1: 1772}, 't50_only_switch_rate': 0.21491813220133416}`

## Interpretation

This is a t+50-only all-agent neural specialist. It uses only past/current all-agent tokens as input; future endpoints are labels/evaluation only. If it does not beat Stage37 while preserving easy cases, all-agent t+50 remains the next blocker.
