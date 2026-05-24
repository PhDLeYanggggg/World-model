# Stage 26 SDD Selector Benchmark

- SDD remains pixel-space raw-frame; no metric or seconds-level claim.
- No latent generative, SMC, JEPA continuation, or ordinary residual training.

| model | t+50 improvement | hard/failure improvement | easy degradation | selector regret | harm over fallback |
| --- | ---: | ---: | ---: | ---: | ---: |
| strongest_baseline | 0.000000 | 0.000000 | 0.000000 | 6.843625 | 0.000000 |
| stage24_selector | -0.432650 | 0.000000 | 11.328798 | 29.636851 | 0.000000 |
| stage25_regret_selector | 0.013573 | 0.010930 | 0.009655 | 6.562431 | -0.281194 |
| stage26_expected_fde_selector | 0.145698 | 0.115397 | 0.024293 | 3.872659 | -2.970966 |
| stage26_failure_assisted_selector | 0.145837 | 0.112341 | 0.018088 | 3.953336 | -2.890289 |
| stage26_conservative_fallback_selector:stage26_failure_assisted_selector | 0.145837 | 0.112341 | 0.018088 | 3.953336 | -2.890289 |

- selected deployment candidate: `{'model': 'stage26_failure_assisted_selector', 't50_improvement': 0.14583655843823773, 'hard_failure_improvement': 0.11234058960663984, 'easy_degradation': 0.01808836280803794, 'selector_regret': 3.9533363458415867, 'harm_over_fallback': -2.8902885153210165}`
- correction skip reason: `Not requested in Stage26; ordinary residual remains forbidden.`
