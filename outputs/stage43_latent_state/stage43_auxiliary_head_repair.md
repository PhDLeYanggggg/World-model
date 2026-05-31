# Stage43-W Auxiliary Density/Validity Head Repair

- source: `fresh_stage43_w_auxiliary_head_repair`
- result_source: `fresh_train_val_selected_auxiliary_head_repair`
- gate: `10 / 10`
- verdict: `stage43_w_density_proxy_repaired_validity_proxy_diagnostic`
- deploy density proxy head: `True`
- deploy true physical validity: `False`

## Density Proxy Repair

- selected feature set: `latent_heads_causal_x`
- l2: `0.1`
- original Stage43-M density R2: `-0.5639`
- repaired density R2: `0.8178`
- repaired density corr: `0.9252`
- RMSE improvement: `0.2708`

## Waypoint Validity Proxy

- selected feature set: `latent_heads_context`
- l2: `0.1`
- original proxy R2: `-2.5067`
- repaired proxy R2: `0.9223`
- deployment status: diagnostic proxy only, not true physical validity

## Interpretation

Stage43-W freezes the Stage43-M latent checkpoint and trains small train/val-selected ridge calibrators for weak auxiliary heads. This repairs the causal history-density proxy if the held-out test R2 is positive, but it is not a future occupancy claim. The validity target remains a waypoint-label availability proxy rather than a verified physical-validity label.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
