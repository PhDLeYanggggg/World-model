# Stage43-X Interaction/Validity Proxy Head Audit

- source: `fresh_stage43_x_interaction_validity_proxy`
- result_source: `fresh_future_label_proxy_head_audit`
- gate: `10 / 10`
- verdict: `stage43_x_interaction_proxy_signal_validity_proxy_diagnostic`
- deploy interaction risk proxy head: `True`
- deploy true physical validity: `False`

## Future Interaction Risk Proxy

- fixed threshold: `0.1`
- selected feature set: `causal_x`
- AUROC: `0.7694`
- AUPRC: `0.3254`
- positive rate: `0.1349`
- ECE: `0.0389`

## Smoothness / Validity Proxy

- selected feature set: `latent_heads_context`
- R2: `0.9216`
- corr: `0.9617`
- RMSE: `0.1222`

## Interpretation

Stage43-X uses future full-waypoints only to construct supervised/evaluation labels. Inputs remain frozen latent/context predictions and causal features. The interaction label is a future-proximity proxy, not a human interaction annotation. The smoothness/validity label is a diagnostic proxy and is not true physical validity.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
