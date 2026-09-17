# Paired Baseline-Relative Parameterizations

All three seeds and complete budgets. Same data, target, metric, policies and CV floor.
Skip versus bounded changes only the residual amplitude mapping. v6 also differs in skip/initialization; its comparison is not a bound-only ablation.

| Seed | Skip gain vs CV % | Bounded gain vs CV % | Bounded vs skip ADE reduction % | Skip oracle % | Bounded oracle % | Skip choice | Bounded choice |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 17 | -0.562987 | 0.075900 | 0.635311 | 0.113319 | 0.161623 | independent | independent |
| 29 | -0.679793 | 0.056063 | 0.730887 | 0.168363 | 0.337019 | independent | independent |
| 43 | -0.552938 | 0.052557 | 0.602165 | 0.093872 | 0.248990 | floor | independent |

## Absolute and Relative Easy Error

| Model | Seed | CV easy ADE | Candidate easy ADE | Easy degradation % | Hard gain % |
| --- | --- | ---: | ---: | ---: | ---: |
| baseline_skip | 17 | 0.00323052 | 0.21272548 | 6484.862511 | -0.040852 |
| baseline_skip | 29 | 0.00323052 | 0.26926540 | 8235.041343 | -0.014158 |
| baseline_skip | 43 | 0.00323052 | 0.21552717 | 6571.588189 | -0.022478 |
| motion_bounded | 17 | 0.00323052 | 0.00525221 | 62.580921 | 0.076554 |
| motion_bounded | 29 | 0.00323052 | 0.01610633 | 398.567330 | 0.110964 |
| motion_bounded | 43 | 0.00323052 | 0.01318268 | 308.066569 | 0.087380 |
| v6_absolute_reference | 17 | 0.00323052 | 1.97733379 | 61107.858785 | -0.615791 |
| v6_absolute_reference | 29 | 0.00323052 | 2.74915869 | 84999.500203 | -0.621934 |
| v6_absolute_reference | 43 | 0.00323052 | 2.36069984 | 72974.856227 | -0.622502 |

## Native-Coordinate Causal Context

| Model | Seed | Recording | Candidate ADE | CV gain % | Gain over best development causal % |
| --- | --- | --- | ---: | ---: | ---: |
| baseline_skip | 17 | ucy_students01 | 0.449504 | 1.887929 | 1.084741 |
| baseline_skip | 17 | ucy_students03 | 0.666174 | 2.291743 | -5.972565 |
| baseline_skip | 29 | ucy_students01 | 0.447139 | 2.404187 | 1.605225 |
| baseline_skip | 29 | ucy_students03 | 0.659298 | 3.300171 | -4.878842 |
| baseline_skip | 43 | ucy_students01 | 0.450992 | 1.563112 | 0.757264 |
| baseline_skip | 43 | ucy_students03 | 0.668528 | 1.946492 | -6.347017 |
| motion_bounded | 17 | ucy_students01 | 0.450009 | 1.777564 | 0.973472 |
| motion_bounded | 17 | ucy_students03 | 0.659550 | 3.263296 | -4.918836 |
| motion_bounded | 29 | ucy_students01 | 0.451063 | 1.547636 | 0.741662 |
| motion_bounded | 29 | ucy_students03 | 0.641869 | 5.856597 | -2.106190 |
| motion_bounded | 43 | ucy_students01 | 0.451891 | 1.366883 | 0.559429 |
| motion_bounded | 43 | ucy_students03 | 0.651560 | 4.435077 | -3.647944 |
| v6_absolute_reference | 17 | ucy_students01 | 0.477761 | -4.279797 | -5.133477 |
| v6_absolute_reference | 17 | ucy_students03 | 0.657150 | 3.615227 | -4.537138 |
| v6_absolute_reference | 29 | ucy_students01 | 0.475305 | -3.743661 | -4.592952 |
| v6_absolute_reference | 29 | ucy_students03 | 0.648099 | 4.942787 | -3.097291 |
| v6_absolute_reference | 43 | ucy_students01 | 0.477596 | -4.243717 | -5.097101 |
| v6_absolute_reference | 43 | ucy_students03 | 0.648122 | 4.939419 | -3.100943 |

Native values remain recording-local and do not replace the primary metric. The development-best causal alternative is descriptive, not a new selected floor.
Oracle uses labels only for diagnosis. Training-seed variation is not a scene CI; one physical development site cannot establish a cross-scene result.
Full family tables retain every learned head/policy and fallback decision. No deployment, metric/seconds, Stage5C or SMC claim.
