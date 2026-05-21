# Stage 5B Gates

Passed: 8 / 10

| gate | pass | evidence | next fix |
| --- | --- | --- | --- |
| Actual Data Conversion Gate | True | 4 actual converted datasets | Convert at least three real datasets. |
| Verified t+100 Gate | True | 2 datasets have actual verified t+100 | Add a second true long-horizon source. |
| No Leakage Gate | True | 4/4 leakage audits passed | Fix split or causal feature flags. |
| Baseline Gate | True | 4 datasets have baseline metrics | Run Stage 5B baseline benchmark. |
| Deterministic Learned Gate | False | learned residual beats strongest causal baseline by >=5% on 1 datasets | Improve deterministic model before latent/SMC. |
| Long-Horizon Gate | False | multistep better on 1/4 comparable datasets | Make multi-step training improve t+50/t+100. |
| Physical Validity Gate | True | linear residual model has no scene projection and reports kinematic validity only | Do not degrade physical validity. |
| Cross-Dataset Gate | True | cross-dataset report exists | Run cross-dataset evaluation. |
| Data Card Gate | True | data_card_stage5b.md exists | Create data cards for converted datasets. |
| Model Card Gate | True | model_card_stage5b.md exists | Create deterministic model card. |

latent_stage5c_ready: `False`
smc_ready: `False`
expert_audit_score: `68`
verdict: `stage5b_usable_data_lake_but_deterministic_gate_failed`
