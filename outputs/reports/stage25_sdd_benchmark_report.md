# Stage 25 SDD Benchmark Report

- SDD remains pixel-space; t+100 remains raw-frame diagnostic.
- Oracle selector is diagnostic only and not counted as real model success.

- best real model: `{'model': 'regret_selector', 't50_improvement': 0.013572603215101897, 't100_raw_frame_improvement': 0.012845981057917899, 'hard_failure_improvement': 0.010929538646649695, 'easy_degradation': 0.00965540295110423, 'harm_over_fallback': -0.28119418047189715, 'selector_regret': 6.562430680690706, 'switch_rate': 0.01625}`
- stage25 beats BPSG-MA v1: `True`
- scene/goal lift: `0.0`
- interaction lift: `0.0`
