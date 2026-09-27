# Metric Direction Note

The registered generic aggregator stores the minimum locality statistic in its
`worst_locality` field. This names the worst gain correctly, but for absolute
harm it is only the minimum, not the worst harm. No harm-minimum field is used
in a gate or a safety conclusion. The correct maxima are shown below without
altering the frozen evaluation file, primary gains, intervals or training.

| Endpoint/subset | Maximum locality mean harm vs legacy |
|---|---:|
| ADE_all | +0.252 |
| ADE_positive_easy | +0.023 |
| ADE_hard | +1.013 |
| ADE_zero_CV | undefined |
| FDE_all | +0.620 |
| FDE_positive_easy | -0.000 |
| FDE_hard | +2.366 |
| FDE_zero_CV | undefined |
