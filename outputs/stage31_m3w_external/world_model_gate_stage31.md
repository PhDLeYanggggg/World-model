# Stage31 Gates

- gates passed: `10 / 11`
- verdict: `stage31_external_domain_gap_sdd_candidate_only`
- Stage5C executed: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| Gate1 external conversion pass or hard blocker | True | rows={'train': 119109, 'val': 7685, 'test': 3636} |
| Gate2 external no-leakage pass | True | causal/no future/test goal checks pass |
| Gate3 external strongest baseline computed | True | baseline table exists |
| Gate4 external latent cache built | True | 15f34cec5c44eff626f12e5109c1350b4f0a3082738d397575458018572660ed |
| Gate5 zero-shot transfer evaluated or blocker | True | zero-shot evaluated |
| Gate6 adapted transfer evaluated if zero-shot fails | True | adapted selector head evaluated |
| Gate7 external improvement positive or domain gap explained | True | zero=-0.9266750268846149, adapted=0.0 |
| Gate8 no metric/seconds overclaim | True | unverified_weak_metric_diagnostic |
| Gate9 world model generalization gate | False | requires positive zero-shot external improvement |
| Gate10 Stage5C false plan only | True | Stage5C not executed |
| Gate11 SMC false | True | SMC not enabled |
