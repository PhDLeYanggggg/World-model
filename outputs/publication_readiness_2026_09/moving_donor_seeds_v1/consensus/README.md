# Fixed Mean And Unanimity Diagnostic

Specified before seed29/43readout. All three augmented risk heads are used;
no threshold, donor, seed or model is selected from these outcomes.

| Context | Arm | Decision cost | Change vs forest % | False-safe cost | Accepted | Harmful |
|---|---|---:|---:|---:|---:|---:|
| critic0 | mean | 0.0032781 | +4.261653 | 1.622e-04 | 442 | 34 |
| critic0 | unanimity | 0.0033505 | +2.146243 | 1.132e-05 | 109 | 7 |
| critic1 | mean | 0.0010090 | +4.080940 | 2.651e-04 | 540 | 134 |
| critic1 | unanimity | 0.0010195 | +3.082273 | 4.248e-05 | 111 | 26 |
| critic2 | mean | 0.0050098 | +3.613417 | 2.256e-04 | 1667 | 323 |
| critic2 | unanimity | 0.0050441 | +2.954254 | 8.873e-06 | 241 | 26 |

The maximum score implements a strict intersection of admissions. It
reduces harmful counts versus the single seed17head46/229/602 to7/26/26,
but also reduces accepted cases. The forest harmful counts are0/1/0.
Both fixed ensemble rules lower total known-proposal cost in all three
contexts relative to the forest; false-safe cost is still higher.

These point comparisons do not establish a full-policy2% risk pass,
coverage of uncertainty estimates, physical safety or independent transfer.
The denominator, unknown-label contribution and reference/support conditions
of the original full-population policy still need evaluation. An accepted-set
intersection alone does not prove that a risk ratio falls. Empty support is NA.
No deployment, identity certificate, new outer C/H, Stage5C or SMC.
