# Observed-Unit Conditioning: Fixed Fit-Only Comparison

27 fresh geometry-only Torch fits; nine cached_verified geometry controls. No held-score selection.
8 observed / 12 predicted native annotation steps. Primary evaluation remains past-normalized ADE, equal physical site.
SDD auxiliary supervision remains 2,000 updates, followed by 4,000 main updates. No sealed roles opened.

| Arm | Gain vs CV (%) | Gain vs legacy (%) | Safe positive fits |
| --- | ---: | ---: | ---: |
| legacy_sdd_aux | -0.805231 | 0.000000 | 0/9 |
| unit_inputs_only | -0.687296 | 0.116993 | 0/9 |
| unit_primary_log | -2.489283 | -1.670600 | 0/9 |
| unit_internal_log | -185.777151 | -183.494366 | 0/9 |

All exposed-site results, including easy harm, are retained. Three-site bootstrap is descriptive, not independent confirmation.
Input repair is not a contribution claim; internal training loss is not a change to the primary evaluation metric.
No new multimodal performance claim, deployment, metric/seconds, Stage5C or SMC.
