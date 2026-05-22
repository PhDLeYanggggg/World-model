# Stage 17 Baseline Selector Report

Selector uses only causal past features and baseline diagnostics; oracle best baseline is a training label, not an inference input.

- trained: `True`
- selector accuracy: `0.582609`
- selector regret: `0.702371`
- official t+50 improvement: `0.081954`
- hard/failure improvement: `0.040700`
- easy degradation: `0.000000`
- choice distribution: `{'constant_position': 89, 'route_corridor_baseline': 26}`
