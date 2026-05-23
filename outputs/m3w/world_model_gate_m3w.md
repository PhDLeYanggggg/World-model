# M3W Gates

- gates passed: `6 / 14`
- Stage5C readiness: `False`
- SMC readiness: `False`
- current verdict: `m3w_small_numpy_fallback_executed_stage26_remains_best_deployable`

| gate | pass | evidence |
| --- | --- | --- |
| Data Gate | True | SDD Stage26 feature store available. |
| No Leakage Gate | True | Feature store audit forbids future/test leakage. |
| JEPA Non-Collapse Gate | False | Full torch JEPA non-collapse is required; NumPy fallback latent variance is diagnostic only. |
| JEPA Downstream Gate | False | Small run did not prove JEPA improves selector/failure over non-JEPA baseline. |
| Transformer Dynamics Gate | False | Torch Transformer dynamics must execute; NumPy hybrid surrogate is diagnostic only. |
| Selector Gate | False | Selector improves strongest baseline or Stage26. |
| Hard/Failure Gate | False | Hard/failure improvement >=10%. |
| Easy Preservation Gate | True | Easy degradation <=2%. |
| Scene/Goal Gate | False | Scene/goal lift not proven; goal labels remain diagnostic. |
| Interaction Gate | True | Interaction risk head has measurable signal. |
| Physical Validity Gate | True | Selected physical baseline only; no residual/correction violates validity. |
| Reproducibility Gate | True | Checkpoint and config-backed run available. |
| Stage5C Plan Readiness | False | Only plan allowed after full gates; not ready and not executed. |
| SMC Readiness | False | SMC remains false. |
