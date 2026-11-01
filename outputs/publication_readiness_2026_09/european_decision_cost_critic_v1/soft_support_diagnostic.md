# Soft Decision Support: Training Diagnostic

Post-hoc SOURCE-only analysis declared during fitting,before own policy readout. No new training or gate/selection change.

All18 context/critic-seed views and444 original record views retained.
444 record views have at least one strictly positive relaxed best-case expected-risk bound.
Constraint counts(all positive harm,easy positive harm,easy net degradation):[444, 444, 444].
The bound relaxes the network to arbitrary per-action logits in its causal capacity box. It ignores shared parameters and moment coupling.
A positive bound means even this larger soft-policy family cannot meet that training constraint at its frozen temperature.
A nonpositive bound does not prove the actual network can attain it. This is not a hard-decoder impossibility result.
The positive reporting floor1e-10 is numerical,not a relaxation of the original2%risk requirement.

|Context/seed|Record views|Any positive relaxed lower bound|Median minimum soft intervention probability|
|---|---:|---:|---:|
|single0_seed43_controller1_dimensionless/17|33|33|0.797614|
|single0_seed43_controller1_dimensionless/29|33|33|0.797614|
|single0_seed43_controller1_dimensionless/43|33|33|0.797614|
|single0_seed43_controller2_dimensionless/17|6|6|0.758866|
|single0_seed43_controller2_dimensionless/29|6|6|0.758866|
|single0_seed43_controller2_dimensionless/43|6|6|0.758866|
|single1_seed43_controller0_dimensionless/17|33|33|0.794871|
|single1_seed43_controller0_dimensionless/29|33|33|0.794871|
|single1_seed43_controller0_dimensionless/43|33|33|0.794871|
|single1_seed43_controller2_dimensionless/17|35|35|0.764907|
|single1_seed43_controller2_dimensionless/29|35|35|0.764907|
|single1_seed43_controller2_dimensionless/43|35|35|0.764907|
|single2_seed43_controller0_dimensionless/17|6|6|0.752903|
|single2_seed43_controller0_dimensionless/29|6|6|0.752903|
|single2_seed43_controller0_dimensionless/43|6|6|0.752903|
|single2_seed43_controller1_dimensionless/17|35|35|0.766040|
|single2_seed43_controller1_dimensionless/29|35|35|0.766040|
|single2_seed43_controller1_dimensionless/43|35|35|0.766040|

128-corner exhaustive synthetic checks agree with fractional-root bisection.
This can diagnose a smooth-support/actual-fallback mismatch,not independent efficacy or a need to relax safety.
All labels stay in SOURCE supervision/diagnosis. No future input,test goal or held confirmation labels.
