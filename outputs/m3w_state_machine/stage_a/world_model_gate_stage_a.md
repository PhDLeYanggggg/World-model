# M3W State Machine Stage A Gates

- gates passed: `6 / 6`
- current verdict: `stage_a_pass_enter_stage_b`
- Stage5C execution: `False`
- SMC enabled: `False`

| gate | pass | evidence |
| --- | --- | --- |
| A1 v2 t+50 Reproduction | True | 0.1686288243790961 > 0.14583655843823773 |
| A2 Hard/Failure Reproduction | True | 0.1336398986813968 > 0.11234058960663984 |
| A3 Easy Preservation | True | 0.01928694490688554 <= 0.02 |
| A4 No Leakage | True | future/test/central velocity inputs remain forbidden |
| A5 Bootstrap CI Exists | True | Stage28 bootstrap CI exists |
| A6 Policy/Schema/Split Frozen | True | policy, schema, feature split, and latent cache hashes recorded |
