# Stage40 Best Model Card

- model family: Stage37-protected neural candidate-ranker world dynamics
- best neural: `Stage40_causal_transformer_candidate_ranker`
- deployment decision: `keep_stage37_selector`
- exceeds Stage37: `False`
- current best deployable is Stage37 selector unless deployment decision is `deploy_stage40_neural`.
- inputs: past-only history, neighbor proxies, train-safe goal prototypes, candidate baseline rollouts, horizon/domain metadata.
- forbidden inputs: future endpoint, central velocity, test endpoint goals.
- coordinate status: dataset-local / unverified weak metric diagnostic.
- Stage5C executed: `False`; SMC enabled: `False`.
