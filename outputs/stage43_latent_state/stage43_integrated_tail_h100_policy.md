# Stage43-U Integrated Tail + H100 Policy

- source: `fresh_stage43_u_integrated_tail_h100_policy`
- result_source: `fresh_integrated_stage43_p_tail_adapter_plus_stage43_t_h100_specialist`
- gate: `15 / 15`
- verdict: `stage43_u_integrated_tail_h100_policy_pass_family_limited`
- deploy integrated policy: `True`
- specialist rows in full test: `1440`

## Integrated Full-Test Metrics

- full-waypoint ADE improvement: `50.28%`
- endpoint FDE improvement: `51.15%`
- t50 full-waypoint ADE improvement: `51.23%`
- t100 raw-frame diagnostic: `0.18%`
- hard/failure ADE improvement: `47.91%`
- easy degradation: `0.00%`
- switch rate: `72.05%`

## Delta vs Stage43-P

- all ADE delta: `0.03%`
- endpoint FDE delta: `-0.01%`
- t50 delta: `0.00%`
- t100 delta: `0.18%`
- hard/failure delta: `0.03%`
- easy degradation delta: `0.00%`

## H100 Source-Stable Slice

- rows: `1440`
- full-waypoint ADE improvement: `2.59%`
- endpoint FDE improvement: `-0.55%`
- hard/failure ADE improvement: `2.59%`
- easy degradation: `0.00%`
- delta vs Stage43-P ADE on slice: `2.59%`

## Bootstrap CI

- bootstrap n: `1000`
- all ADE CI: `[49.97%, 50.58%]`
- t100 diagnostic CI: `[0.14%, 0.22%]`

## Interpretation

Stage43-U composes the Stage43-P protected full-waypoint tail adapter with the Stage43-T source-stable h100 specialist. It adds a small positive h100 family-limited full-waypoint ADE lift while preserving Stage43-P t50 and easy-case safety. Endpoint FDE on the h100 slice remains negative, so this is not a uniform t100 or endpoint-success claim.

Claim boundary unchanged: dataset-local/raw-frame 2.5D only; no metric/seconds-level claim; no Stage5C execution; no SMC.
