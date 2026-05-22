# Stage 17 Benchmark Report

Compared models: global strongest causal baseline, per-sample oracle baseline diagnostic, trained selector, BPSG-MA v1, selector-only, selector+correction, and no-scene/no-goal/no-interaction selector ablations.

- official FDE@50 selector improvement: `0.081954`
- diagnostic FDE@100 oracle improvement: `0.106599`
- selector regret: `0.702371`
- hard/failure improvement: `0.040700`
- easy degradation: `0.000000`

The oracle selector has clear headroom, but the trained selector does not clear Stage17 gates.
