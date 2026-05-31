# Stage43-Q T100 Guarded Trial

- source: `fresh_stage43_q_t100_guarded_trial`
- result_source: `fresh_validation_selected_t100_guarded_trial`
- gate: `11 / 11`
- verdict: `stage43_q_t100_guarded_trial_honest_blocker`
- full-test rows: `89736`
- t100 status: `honest_blocker_no_t100_deployment`
- t100 blocker: `validation_positive_h100_did_not_generalize_to_test_safely`

## Selected Validation Trial

- target: `residual`
- train filter: `t50t100`
- l2: `1000.0`
- min h100 validation improvement: `0.00%`
- train rows: `58845`
- h100 allowed rules: `UCY|100`

## Full-Test Metrics

- full-waypoint ADE improvement: `50.25%`
- endpoint FDE improvement: `51.15%`
- t50 full-waypoint ADE improvement: `51.23%`
- t100 raw-frame diagnostic: `0.00%`
- hard/failure full-waypoint ADE improvement: `47.88%`
- easy degradation: `0.00%`
- switch rate: `70.45%`

## Rejected H100 Candidate Test Metrics

- candidate full-waypoint ADE improvement: `49.91%`
- candidate t100 raw-frame diagnostic: `-2.22%`
- candidate hard/failure improvement: `47.47%`
- candidate easy degradation: `0.00%`
- candidate t100 delta vs Stage43-P: `-2.22%`

## Delta vs Stage43-P

- all ADE delta: `0.00%`
- t50 delta: `0.00%`
- t100 delta: `0.00%`
- hard/failure delta: `0.00%`
- easy degradation delta: `0.00%`

## Bootstrap CI

- bootstrap n: `1000`
- all ADE CI: `[49.97%, 50.55%]`
- t50 ADE CI: `[50.72%, 51.69%]`
- t100 ADE CI: `[0.00%, 0.00%]`
- hard/failure ADE CI: `[47.55%, 48.23%]`

## Rejected H100 Candidate Family Test Table

| family|horizon | rows | ADE lift | delta vs Stage43-P | easy degradation | switch |
| --- | ---: | ---: | ---: | ---: | ---: |
| TrajNet_biwi|100 | 1160 | 0.00% | 0.00% | 0.00% | 0.00% |
| TrajNet_crowds|100 | 1440 | 0.00% | 0.00% | 0.00% | 0.00% |
| UCY|100 | 15470 | -2.49% | -2.49% | 18.84% | 100.00% |

## Interpretation

Stage43-Q is a validation-selected h100 add-on trial over the Stage43-P safety floor. It does not change the already deployed h10/h25/h50 rules. If validation-selected h100 support is weak or fails on test, the deployment remains Stage43-P and t100 remains an honest raw-frame diagnostic blocker.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
