# Stage43-V World-State Head Audit

- source: `fresh_stage43_v_world_state_head_audit`
- result_source: `fresh_checkpoint_replay_world_state_head_audit`
- gate: `9 / 9`
- verdict: `stage43_v_world_state_head_audit_partial`
- test rows: `89736`

## Auxiliary Head Metrics

| head | primary metric | calibration/error | note |
| --- | ---: | ---: | --- |
| failure | AUROC `0.8648`, AUPRC `0.7901` | ECE `0.1254` | baseline-failure risk label |
| gain | AUROC `0.8737`, AUPRC `0.9215` | ECE `0.1121` | switch/gain opportunity label |
| harm | AUROC `0.9047`, AUPRC `0.8891` | ECE `0.1330` | easy/harm guard label |
| density | R2 `-0.5639`, corr `0.2055` | RMSE `0.4111` | occupancy-density proxy |
| physical_validity_proxy | R2 `-2.5067` | RMSE `0.4184` | not trained with explicit loss; not deployable |

## Latent State

- latent dim: `32`
- mean variance: `0.482653`
- min variance: `0.108561`
- non-collapse threshold: `0.01`

## Interpretation

Stage43-V audits the auxiliary world-state heads from the existing Stage43-M latent dynamics checkpoint. This is evidence about latent risk/density heads, not a new deployment policy. The physical-validity output is explicitly marked not deployable because the current training loss does not supervise it.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
