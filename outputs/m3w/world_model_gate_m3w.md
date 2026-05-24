# M3W Gates

- gates passed: `12 / 16`
- Stage5C readiness: `False`
- SMC readiness: `False`
- current verdict: `m3w_stage27_evidence_executed_not_ccfa_candidate_stage26_remains_best`

| gate | pass | evidence |
| --- | --- | --- |
| Data Gate | True | SDD Stage26 feature store available. |
| No Leakage Gate | True | Feature store audit forbids future/test leakage. |
| JEPA Non-Collapse Gate | False | Full torch JEPA latent variance must be non-collapsed; current small run did not pass. |
| JEPA Downstream Gate | True | Hybrid with JEPA features should improve over Transformer-only in the evidence matrix. |
| Transformer Dynamics Gate | True | Torch Transformer dynamics variant executed and was validation-selected. |
| Hybrid Gate | True | Hybrid should beat JEPA-only or Transformer-only on the evidence matrix. |
| Selector Gate | True | Selector improves strongest baseline or Stage26. |
| Hard/Failure Gate | True | Hard/failure improvement >=10%. |
| Easy Preservation Gate | True | Easy degradation <=2%. |
| Scene/Goal Gate | False | Scene/goal lift not proven; goal labels remain diagnostic. |
| Interaction Gate | True | Interaction risk head has measurable signal. |
| Physical Validity Gate | True | Selected physical baseline only; no residual/correction violates validity. |
| Reproducibility Gate | True | Checkpoint and config-backed run available. |
| CCF-A Evidence Gate | True | Paper package, ablation table, and bootstrap report exist; passing this gate does not mean CCF-A quality is reached. |
| Stage5C Plan Readiness | False | Only plan allowed after full gates; not ready and not executed. |
| SMC Readiness | False | SMC remains false. |
