# Stage43-Y Multimodal Latent Head Suite

- source: `fresh_stage43_y_multimodal_latent_head_suite`
- result_source: `fresh_consolidated_stage43_vwx_head_suite`
- gate: `12 / 12`
- verdict: `stage43_y_protected_multimodal_latent_head_suite_candidate`
- protected multimodal latent state candidate: `True`
- standalone ungated deployment: `False`

## Latent State

- latent dim: `32`
- min variance: `0.108561`
- mean variance: `0.482653`
- non-collapse threshold: `0.01`

## Head Suite

| head | status | primary metric | note |
| --- | --- | ---: | --- |
| failure_risk | `deployable_proxy` | AUROC `0.8648` | baseline failure risk |
| gain_opportunity | `deployable_proxy` | AUROC `0.8737` | switch/gain opportunity |
| harm_guard | `deployable_proxy` | AUROC `0.9047` | easy/harm guard |
| causal_history_density | `deployable_proxy_not_future_occupancy` | R2 `0.8178` | causal history-density proxy |
| future_interaction_risk | `deployable_proxy_not_human_annotation` | AUROC `0.7694` | future-proximity interaction risk proxy |
| waypoint_label_availability | `diagnostic_only` | R2 `0.9223` | waypoint label availability proxy |
| smoothness_validity_proxy | `diagnostic_only_not_true_physical_validity` | R2 `0.9216` | future waypoint smoothness/validity proxy |

## Deployment Contract

- deployable proxy heads: `failure_risk`, `gain_opportunity`, `harm_guard`, `causal_history_density`, `future_interaction_risk`
- diagnostic-only heads: `waypoint_label_availability`, `smoothness_validity_proxy`
- safety floor remains required
- this is not a standalone ungated deployment policy
- no true physical-validity claim
- no future occupancy claim

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
