# Distance And Source Diagnostics

All bins were fixed before readout. Distances are standardized causal
feature-space distances, not metres or calibrated semantic similarity.
Unsafe here means at least one positive signed risk target, not collision.
The following source/query-weighted fractions are descriptive. They do
not establish independent calibration or irreducible uncertainty.

| Source | Nearest-distance interval | Rows | Predicted unsafe % | Observed unsafe % |
|---|---|---:|---:|---:|
| moving_source_component_048 | [0.0, 0.25) | 0 | NA | NA |
| moving_source_component_048 | [0.25, 0.5) | 79 | +2.484903 | +9.652869 |
| moving_source_component_048 | [0.5, 1.0) | 1329 | +3.477019 | +13.499017 |
| moving_source_component_048 | [1.0, 2.0) | 355 | +5.119268 | +19.844576 |
| moving_source_component_048 | [2.0, 4.0) | 6 | +5.524908 | +21.341361 |
| moving_source_component_048 | [4.0, inf) | 0 | NA | NA |
| moving_source_component_067 | [0.0, 0.25) | 0 | NA | NA |
| moving_source_component_067 | [0.25, 0.5) | 59 | +12.502841 | +1.384209 |
| moving_source_component_067 | [0.5, 1.0) | 2652 | +15.540585 | +4.426509 |
| moving_source_component_067 | [1.0, 2.0) | 1274 | +17.170601 | +5.880943 |
| moving_source_component_067 | [2.0, 4.0) | 13 | +17.354195 | +14.199942 |
| moving_source_component_067 | [4.0, inf) | 0 | NA | NA |
| moving_source_component_053 | [0.0, 0.25) | 0 | NA | NA |
| moving_source_component_053 | [0.25, 0.5) | 85 | +28.320548 | +8.374352 |
| moving_source_component_053 | [0.5, 1.0) | 482 | +34.971466 | +18.176237 |
| moving_source_component_053 | [1.0, 2.0) | 102 | +58.235894 | +36.168854 |
| moving_source_component_053 | [2.0, 4.0) | 3 | +74.989260 | +29.166667 |
| moving_source_component_053 | [4.0, inf) | 0 | NA | NA |
| moving_source_component_065 | [0.0, 0.25) | 0 | NA | NA |
| moving_source_component_065 | [0.25, 0.5) | 1 | +31.907238 | +0.000000 |
| moving_source_component_065 | [0.5, 1.0) | 41 | +35.126199 | +21.645022 |
| moving_source_component_065 | [1.0, 2.0) | 2 | +31.897500 | +0.000000 |
| moving_source_component_065 | [2.0, 4.0) | 0 | NA | NA |
| moving_source_component_065 | [4.0, inf) | 0 | NA | NA |
| moving_source_component_076 | [0.0, 0.25) | 0 | NA | NA |
| moving_source_component_076 | [0.25, 0.5) | 845 | +6.534774 | +31.409490 |
| moving_source_component_076 | [0.5, 1.0) | 1893 | +8.242634 | +48.333585 |
| moving_source_component_076 | [1.0, 2.0) | 249 | +16.049236 | +68.233182 |
| moving_source_component_076 | [2.0, 4.0) | 12 | +26.991853 | +84.923372 |
| moving_source_component_076 | [4.0, inf) | 0 | NA | NA |
| moving_source_component_059 | [0.0, 0.25) | 2 | +12.892729 | +0.000000 |
| moving_source_component_059 | [0.25, 0.5) | 825 | +22.913072 | +19.688441 |
| moving_source_component_059 | [0.5, 1.0) | 2668 | +29.313592 | +27.222795 |
| moving_source_component_059 | [1.0, 2.0) | 3667 | +40.762121 | +37.009225 |
| moving_source_component_059 | [2.0, 4.0) | 1610 | +46.836406 | +28.768822 |
| moving_source_component_059 | [4.0, inf) | 0 | NA | NA |
| moving_source_component_060 | [0.0, 0.25) | 508 | +24.361331 | +33.330981 |
| moving_source_component_060 | [0.25, 0.5) | 5782 | +27.065248 | +33.160826 |
| moving_source_component_060 | [0.5, 1.0) | 3977 | +32.187111 | +38.371825 |
| moving_source_component_060 | [1.0, 2.0) | 481 | +33.807982 | +65.168335 |
| moving_source_component_060 | [2.0, 4.0) | 1 | +58.130711 | +100.000000 |
| moving_source_component_060 | [4.0, inf) | 0 | NA | NA |
| moving_source_component_073 | [0.0, 0.25) | 2557 | +31.094075 | +31.383366 |
| moving_source_component_073 | [0.25, 0.5) | 16121 | +31.996605 | +40.877891 |
| moving_source_component_073 | [0.5, 1.0) | 379 | +36.454560 | +33.632577 |
| moving_source_component_073 | [1.0, 2.0) | 1 | +21.690611 | +0.000000 |
| moving_source_component_073 | [2.0, 4.0) | 0 | NA | NA |
| moving_source_component_073 | [4.0, inf) | 0 | NA | NA |

No nearest-distance cutoff is selected from this table. Empty bins stay NA.
Source076 underestimates signed-risk exceedance even in the [0.25,0.5)
and [0.5,1) bins; excluding only very distant histories would not explain
away the observed failures. Source067 instead overestimates the probability
of exceedance in those bins. A single global distance-to-risk interpretation
is not supported. Probability and harm severity remain different quantities.

Mixed neighbor labels are common, but 32 references make mixing likely.
This does not prove that the future is fundamentally unpredictable.
