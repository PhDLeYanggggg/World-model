# All Cutoff-Relative Risk Controls

Four already design-exposed SDD sites, three seeds. No deployment selection.
Obs8/pred12 stride12 annotation pixels, not raw t50 or seconds. Gains are over CV, not a strongest-baseline claim.

| Action | Policy | ADE gain % | FDE gain % | Hard gain % | Worst positive-easy degradation % | Zero-CV harms, row/seed | Mean selected |
|---|---|---:|---:|---:|---:|---:|---:|
| damped_velocity_005 | floor | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0 | 0.0 |
| damped_velocity_005 | uncontrolled | -6.621048 | -12.182784 | 8.003347 | 134.469180 | 21 | 175756.0 |
| damped_velocity_005 | old_strict | 3.634683 | 4.649955 | 4.723212 | 2.494391 | 0 | 21533.7 |
| damped_velocity_005 | native_point | 0.626177 | 0.852485 | 0.076972 | -0.617962 | 0 | 10905.7 |
| damped_velocity_005 | native_population | 1.401053 | 1.755998 | 1.326697 | 0.115543 | 0 | 15602.3 |
| damped_velocity_005 | native_selected | 0.646430 | 0.880673 | 0.094897 | -0.580861 | 0 | 11209.3 |
| damped_velocity_005 | dimensionless_point | 1.343867 | 1.804054 | 1.082791 | -0.083251 | 0 | 13663.0 |
| damped_velocity_005 | dimensionless_population | 2.011759 | 2.626055 | 2.132084 | 1.264676 | 1 | 18831.3 |
| damped_velocity_005 | dimensionless_selected | 1.359893 | 1.826368 | 1.102011 | -0.012390 | 0 | 13908.3 |
| damped_velocity_005 | dimensionless_matched_native_population | 1.130634 | 1.462020 | 1.009391 | 0.373273 | 0 | 12984.7 |
| damped_velocity_005 | cutoff_point | 0.646366 | 0.880692 | 0.100538 | -0.591840 | 0 | 11045.3 |
| damped_velocity_005 | cutoff_population | 1.432180 | 1.795960 | 1.398032 | 0.135072 | 0 | 15715.3 |
| damped_velocity_005 | cutoff_selected | 0.667723 | 0.908936 | 0.120303 | -0.569173 | 0 | 11345.0 |
| transformer | floor | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0 | 0.0 |
| transformer | uncontrolled | 7.633081 | 8.645107 | 10.657509 | 32.743338 | 9000 | 175756.0 |
| transformer | old_strict | 2.436827 | 2.694480 | 2.291813 | 1.066900 | 0 | 11732.7 |
| transformer | native_point | 1.655377 | 1.717837 | 0.838074 | -0.797599 | 0 | 14153.0 |
| transformer | native_population | 3.574607 | 3.856414 | 3.044722 | 1.243031 | 7 | 44410.3 |
| transformer | native_selected | 1.753023 | 1.825806 | 0.884873 | -0.785465 | 0 | 17520.3 |
| transformer | dimensionless_point | 2.883486 | 3.037838 | 2.724271 | 0.910340 | 0 | 17762.0 |
| transformer | dimensionless_population | 4.988086 | 5.405931 | 5.252471 | 4.450332 | 5 | 51630.0 |
| transformer | dimensionless_selected | 2.986166 | 3.154557 | 2.791706 | 0.943656 | 0 | 21355.0 |
| transformer | dimensionless_matched_native_population | 3.835997 | 4.122220 | 4.007214 | 3.320932 | 5 | 37824.7 |
| transformer | cutoff_point | 1.673652 | 1.735010 | 0.852618 | -0.773232 | 1 | 14554.0 |
| transformer | cutoff_population | 3.604563 | 3.887128 | 3.082321 | 1.273190 | 5 | 44673.7 |
| transformer | cutoff_selected | 1.771752 | 1.845245 | 0.899853 | -0.750879 | 1 | 17919.3 |
| eqmotion | floor | 0.000000 | 0.000000 | 0.000000 | -0.000000 | 0 | 0.0 |
| eqmotion | uncontrolled | 11.043498 | 12.391090 | 15.401484 | 47.391895 | 34584 | 175756.0 |
| eqmotion | old_strict | 1.609305 | 1.730640 | 0.527132 | 0.452161 | 0 | 6901.0 |
| eqmotion | native_point | 1.505354 | 1.618932 | 0.143881 | -0.469674 | 0 | 8629.0 |
| eqmotion | native_population | 3.297619 | 3.483648 | 2.028601 | 1.345792 | 7 | 28395.7 |
| eqmotion | native_selected | 1.589097 | 1.703610 | 0.161337 | -0.417987 | 0 | 9499.3 |
| eqmotion | dimensionless_point | 3.758440 | 4.096065 | 3.093255 | 3.525947 | 0 | 11324.7 |
| eqmotion | dimensionless_population | 5.972731 | 6.439929 | 5.796096 | 9.009714 | 6 | 35819.3 |
| eqmotion | dimensionless_selected | 3.801128 | 4.132934 | 3.122710 | 3.529848 | 0 | 12091.3 |
| eqmotion | dimensionless_matched_native_population | 3.095356 | 3.297811 | 2.363647 | 3.836807 | 3 | 22669.3 |
| eqmotion | cutoff_point | 1.541348 | 1.656685 | 0.160853 | -0.366782 | 0 | 8734.0 |
| eqmotion | cutoff_population | 3.337008 | 3.522162 | 2.053225 | 1.458027 | 6 | 28666.7 |
| eqmotion | cutoff_selected | 1.623660 | 1.742368 | 0.176799 | -0.364199 | 0 | 9608.0 |

The 30 old controls reproduce exactly; they are cached_verified model/decision evidence with fresh reduction checks.
The previous failed matched-count controls remain failures. Three new rules do not establish equal-coverage superiority.
Zero-CV cases are not silently divided by zero. Unknown-label selections and full-grid bounds remain in analysis.json.

## All Registered Paired Contrasts

3000 paired four-physical-site resamples, nominal conditional development intervals; not independent confirmation.

### damped_velocity_005__cutoff_point_minus_native_point

```json
{
  "all": {
    "mean_gain_difference_pp": 0.0201896836641402,
    "scene_differences_pp": [
      0.038939557572581496,
      0.023607075531084654,
      0.006323749255554567,
      0.011888352297340088
    ],
    "ci95_pp": [
      0.009106050776447328,
      0.032176756253771144
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.02356507005784325,
    "scene_differences_pp": [
      0.053529829344722835,
      0.0258666488461734,
      -0.00021980587735503931,
      0.01508360791783181
    ],
    "ci95_pp": [
      0.006301807803527071,
      0.04391827398800008
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.00808545363558677,
    "scene_differences_pp": [
      0.009895514989066712,
      -0.005313233155035402,
      -0.023756228470683816,
      -0.013167867905694575
    ],
    "ci95_pp": [
      -0.019145479641771712,
      0.00412966926537639
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__cutoff_point_minus_dimensionless_point

```json
{
  "all": {
    "mean_gain_difference_pp": -0.6975008480914507,
    "scene_differences_pp": [
      -0.22531741563261987,
      -1.4036627661942136,
      -0.46619266707706997,
      -0.6948305434618995
    ],
    "ci95_pp": [
      -1.1692952414149276,
      -0.3426956975899398
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -0.9822530768344967,
    "scene_differences_pp": [
      -0.44104119360786065,
      -1.7101060397905932,
      -0.6649083874417872,
      -1.1129566864977458
    ],
    "ci95_pp": [
      -1.4488066267033917,
      -0.5529747905248239
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 0.44125899371724453,
    "scene_differences_pp": [
      0.10312002411005938,
      0.6206566149741022,
      0.5343642616274069,
      0.5068950741574096
    ],
    "ci95_pp": [
      0.21093108348939626,
      0.592216229769929
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__cutoff_point_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -2.9883164579269565,
    "scene_differences_pp": [
      -1.871259030349648,
      -4.835743337611486,
      -1.9423889134025707,
      -3.3038745503441214
    ],
    "ci95_pp": [
      -4.112404731559257,
      -1.9068239718761093
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -4.622674666710832,
    "scene_differences_pp": [
      -3.811212289950716,
      -6.111180580299269,
      -2.88283446210561,
      -5.685471334487735
    ],
    "ci95_pp": [
      -5.898325957393503,
      -3.347023376028163
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 2.3760361274540927,
    "scene_differences_pp": [
      1.4367710329284744,
      3.191411913976505,
      2.9326068233161307,
      1.94335473959526
    ],
    "ci95_pp": [
      1.6900628862618672,
      3.0620093686463177
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__cutoff_population_minus_native_population

```json
{
  "all": {
    "mean_gain_difference_pp": 0.03112674807800586,
    "scene_differences_pp": [
      0.09332545904823242,
      0.010357188461351807,
      -0.031600677105203534,
      0.052425021907642755
    ],
    "ci95_pp": [
      -0.010621744321925863,
      0.07287524047793759
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.07133475991844274,
    "scene_differences_pp": [
      0.2013743511819266,
      0.00982355861496309,
      -0.035326874792784846,
      0.10946800466966611
    ],
    "ci95_pp": [
      -0.012751658088910878,
      0.15542117792579635
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.010948147338346592,
    "scene_differences_pp": [
      -0.056586247836909465,
      0.04124145148880487,
      -0.004847418575160134,
      -0.02360037443012164
    ],
    "ci95_pp": [
      -0.04365154052147213,
      0.025030995009073242
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__cutoff_population_minus_dimensionless_population

```json
{
  "all": {
    "mean_gain_difference_pp": -0.5795792430747515,
    "scene_differences_pp": [
      0.37737432608307975,
      -1.6484583711142364,
      -0.4301695880776224,
      -0.6170633391902269
    ],
    "ci95_pp": [
      -1.3438861753550828,
      0.12876490976475308
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -0.7340515072799536,
    "scene_differences_pp": [
      0.7159736455279675,
      -2.0801122941925154,
      -0.6510683704898268,
      -0.9209990099654397
    ],
    "ci95_pp": [
      -1.7228513132668433,
      0.3067304816546157
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 0.6416358053825261,
    "scene_differences_pp": [
      -0.5156535469759693,
      1.0796603019160411,
      0.8992796669537118,
      1.1032567996363207
    ],
    "ci95_pp": [
      -0.11682508475296671,
      1.091458550776181
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__cutoff_population_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -2.2025026311678433,
    "scene_differences_pp": [
      -0.5904601492495387,
      -4.238393618085089,
      -1.6876855468925034,
      -2.2934712104442423
    ],
    "ci95_pp": [
      -3.6007166002869426,
      -1.0162129145482146
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -3.32518001015416,
    "scene_differences_pp": [
      -1.351520019585728,
      -5.438727742976157,
      -2.519176743988416,
      -3.991295534066341
    ],
    "ci95_pp": [
      -4.715011638521249,
      -1.935348381787072
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.799817300455278,
    "scene_differences_pp": [
      0.38586323995660754,
      2.8524540706889256,
      2.725298119176811,
      1.2356537719987681
    ],
    "ci95_pp": [
      0.8107585059776878,
      2.788876094932868
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__cutoff_selected_minus_native_selected

```json
{
  "all": {
    "mean_gain_difference_pp": 0.02129316543345028,
    "scene_differences_pp": [
      0.037216951977636636,
      0.022518912286895176,
      0.00918139864812817,
      0.01625539882114113
    ],
    "ci95_pp": [
      0.012515777057819921,
      0.03197656368851276
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.025405769445852044,
    "scene_differences_pp": [
      0.04982137932014741,
      0.025732594926575203,
      0.00033653624181839703,
      0.02573256729486717
    ],
    "ci95_pp": [
      0.0066855509130075985,
      0.04379917631382735
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.007441202381688883,
    "scene_differences_pp": [
      0.0054952637649763325,
      -0.011328735571602966,
      -0.017119479800709048,
      -0.00681185791941985
    ],
    "ci95_pp": [
      -0.014542574330386748,
      0.001289263930831508
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__cutoff_selected_minus_dimensionless_selected

```json
{
  "all": {
    "mean_gain_difference_pp": -0.692170191913577,
    "scene_differences_pp": [
      -0.20704308215824252,
      -1.4072794570998082,
      -0.4584646505757184,
      -0.6958935778205388
    ],
    "ci95_pp": [
      -1.1700757554687857,
      -0.3292557060738166
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -0.9817080657822663,
    "scene_differences_pp": [
      -0.4278790267321697,
      -1.7171384900167896,
      -0.6569397771847951,
      -1.1248749691953108
    ],
    "ci95_pp": [
      -1.4520888118087911,
      -0.5424094019584824
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 0.4617409185524862,
    "scene_differences_pp": [
      0.11073618617204462,
      0.6775961085260351,
      0.5330932480353101,
      0.525538131476555
    ],
    "ci95_pp": [
      0.216325451637861,
      0.6395816142636651
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__cutoff_selected_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -2.966959672982316,
    "scene_differences_pp": [
      -1.8354804952450676,
      -4.818614498770135,
      -1.9350496066384504,
      -3.2786940912756135
    ],
    "ci95_pp": [
      -4.097723275737214,
      -1.8852650509417592
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -4.602909677953779,
    "scene_differences_pp": [
      -3.7788723725029616,
      -6.098310373287286,
      -2.8789912880756052,
      -5.655464677949261
    ],
    "ci95_pp": [
      -5.876887525618274,
      -3.3289318302892834
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 2.3723371816108396,
    "scene_differences_pp": [
      1.4390814188813272,
      3.1905083682499336,
      2.9411042133631327,
      1.918654725948965
    ],
    "ci95_pp": [
      1.6788680724151461,
      3.065806290806533
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__cutoff_point_minus_native_point

```json
{
  "all": {
    "mean_gain_difference_pp": 0.018275309648571603,
    "scene_differences_pp": [
      0.023887650512044445,
      -0.011247076769649311,
      0.03977459412930484,
      0.020686070722586436
    ],
    "ci95_pp": [
      -0.002463394949225872,
      0.03500246327762524
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.014543816178042013,
    "scene_differences_pp": [
      -0.0011251168271830814,
      -0.013303118997953778,
      0.047643912757278084,
      0.024959587780026826
    ],
    "ci95_pp": [
      -0.00721411791256843,
      0.036301750268652455
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 0.031053029623540063,
    "scene_differences_pp": [
      -0.04031009386062934,
      -0.04510461978413094,
      0.21484430195790605,
      -0.005217469818985521
    ],
    "ci95_pp": [
      -0.04270735682238014,
      0.1510557030032722
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__cutoff_point_minus_dimensionless_point

```json
{
  "all": {
    "mean_gain_difference_pp": -1.2098333207447602,
    "scene_differences_pp": [
      -0.4494776958701152,
      -1.2067645319618014,
      -1.410515632414966,
      -1.7725754227321588
    ],
    "ci95_pp": [
      -1.6311227000395694,
      -0.6897371800063279
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -1.8716525601528866,
    "scene_differences_pp": [
      -0.6330098191768774,
      -1.5000962719911248,
      -2.009609589733885,
      -3.3438945597096597
    ],
    "ci95_pp": [
      -2.882944987780026,
      -0.9771597618161293
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.1669805572834728,
    "scene_differences_pp": [
      0.4097028430968592,
      0.43257855618149543,
      2.331446705238749,
      1.4941941246167878
    ],
    "ci95_pp": [
      0.42114069963917733,
      1.9128204149277686
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__cutoff_point_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -0.7631743848411421,
    "scene_differences_pp": [
      0.2711105020767657,
      -3.0171478609546054,
      -0.43445447608425614,
      0.12779429559752753
    ],
    "ci95_pp": [
      -2.230912321816572,
      0.1994523988371466
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -1.439194880089517,
    "scene_differences_pp": [
      -1.151283681966897,
      -3.803718284709934,
      -0.7459987514722877,
      -0.05577880220895004
    ],
    "ci95_pp": [
      -3.039288401400522,
      -0.3296550221484368
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.6798006312507163,
    "scene_differences_pp": [
      3.1140163717249254,
      1.6320084483106956,
      1.3508334979259407,
      0.6223442070413032
    ],
    "ci95_pp": [
      0.8747602673586513,
      2.6732206532751794
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__cutoff_population_minus_native_population

```json
{
  "all": {
    "mean_gain_difference_pp": 0.029956759183088932,
    "scene_differences_pp": [
      0.12738542556390886,
      -0.040139533181782205,
      0.016371222941757857,
      0.016209921408471217
    ],
    "ci95_pp": [
      -0.02601184415089719,
      0.09959154952504945
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.03759938322776801,
    "scene_differences_pp": [
      0.16016777425884587,
      -0.055320888990639805,
      0.020705869574300095,
      0.024844778068565887
    ],
    "ci95_pp": [
      -0.03527947222583838,
      0.12530229808770943
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.016559236068267036,
    "scene_differences_pp": [
      -0.049479025616328,
      -0.0669230068940041,
      0.042096794218127176,
      0.008068294019136779
    ],
    "ci95_pp": [
      -0.05820101625516605,
      0.025082544118631978
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__cutoff_population_minus_dimensionless_population

```json
{
  "all": {
    "mean_gain_difference_pp": -1.383522542731189,
    "scene_differences_pp": [
      -0.04583760043727114,
      -1.4857694171416913,
      -1.7512249228232912,
      -2.2512582305225024
    ],
    "ci95_pp": [
      -2.0598860271772996,
      -0.47218443103377616
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -2.1701492966194253,
    "scene_differences_pp": [
      -0.015656970853251018,
      -1.8100546432412767,
      -2.6417668871366096,
      -4.213118685246565
    ],
    "ci95_pp": [
      -3.612352674745243,
      -0.6721844499240907
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.714401589613049,
    "scene_differences_pp": [
      0.14849556574393352,
      0.487500964431109,
      3.676332331734744,
      2.5452774965424085
    ],
    "ci95_pp": [
      0.31799826508752127,
      3.1108049141385763
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__cutoff_population_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 1.167736630482763,
    "scene_differences_pp": [
      3.380077428882733,
      -1.6174736244010002,
      0.27290240494804996,
      2.635440312501269
    ],
    "ci95_pp": [
      -0.6722856097264751,
      3.007758870692001
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.7905081442547561,
    "scene_differences_pp": [
      2.9766400082164135,
      -2.360397835612593,
      -0.16661517174713092,
      2.712405576162335
    ],
    "ci95_pp": [
      -1.263506503679862,
      2.844522792189374
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.18108356783220203,
    "scene_differences_pp": [
      -0.08558995603358266,
      0.8233503730870728,
      0.06229189384460154,
      -1.5243865822268998
    ],
    "ci95_pp": [
      -1.1277169632090245,
      0.596115290806909
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__cutoff_selected_minus_native_selected

```json
{
  "all": {
    "mean_gain_difference_pp": 0.018728642067261658,
    "scene_differences_pp": [
      0.025843454408791544,
      -0.017242140063111755,
      0.04408674282634362,
      0.022226511097023227
    ],
    "ci95_pp": [
      -0.00647074144513593,
      0.03862168489401352
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.014980552397422797,
    "scene_differences_pp": [
      0.0015168625701389438,
      -0.016737380567100058,
      0.04858563726833909,
      0.026557090318313215
    ],
    "ci95_pp": [
      -0.007610258998480557,
      0.03757136379332615
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 0.04002138843724168,
    "scene_differences_pp": [
      -0.020777627602731386,
      -0.046175896053624044,
      0.2315121316854185,
      -0.004473054280096367
    ],
    "ci95_pp": [
      -0.035750185610242124,
      0.16843969186338104
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__cutoff_selected_minus_dimensionless_selected

```json
{
  "all": {
    "mean_gain_difference_pp": -1.2144143317211697,
    "scene_differences_pp": [
      -0.44964883073782547,
      -1.2298035743662283,
      -1.3959720429394862,
      -1.7822328788411386
    ],
    "ci95_pp": [
      -1.6441255527224112,
      -0.6862296337882406
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -1.8918526423534257,
    "scene_differences_pp": [
      -0.6484669879824878,
      -1.5245091344727402,
      -2.014319269679854,
      -3.3801151772786198
    ],
    "ci95_pp": [
      -2.91621366657715,
      -0.9899300584068293
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.1835382134120274,
    "scene_differences_pp": [
      0.45287206300470473,
      0.40309946678582786,
      2.414695282897794,
      1.4634860409597827
    ],
    "ci95_pp": [
      0.4279857648952663,
      1.9390906619287884
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__cutoff_selected_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -0.6650749788161853,
    "scene_differences_pp": [
      0.5036795316401221,
      -2.983384421803814,
      -0.4082122664892718,
      0.2276172413882227
    ],
    "ci95_pp": [
      -2.180634006005805,
      0.3656483865141724
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -1.3919598398329098,
    "scene_differences_pp": [
      -1.0517913264712742,
      -3.773884524910398,
      -0.7404304081692903,
      -0.0017330997806763726
    ],
    "ci95_pp": [
      -3.0155209957251214,
      -0.2642476564533258
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.6598116060837778,
    "scene_differences_pp": [
      3.06330395237806,
      1.6330299408369164,
      1.340260544168037,
      0.6026519869520963
    ],
    "ci95_pp": [
      0.8602464754233013,
      2.632543100325554
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__cutoff_point_minus_native_point

```json
{
  "all": {
    "mean_gain_difference_pp": 0.03599463879672693,
    "scene_differences_pp": [
      0.013381028365788694,
      0.07926881077144898,
      0.08460112507879813,
      -0.03327240902912809
    ],
    "ci95_pp": [
      -0.009945690331669699,
      0.08193496792512356
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.01697218950553747,
    "scene_differences_pp": [
      -0.007286333659251021,
      0.06174143943167287,
      0.07045224752156853,
      -0.0570185952718405
    ],
    "ci95_pp": [
      -0.03215246446554576,
      0.0660968434766207
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.037308244920186695,
    "scene_differences_pp": [
      -0.025332347889905904,
      -0.09583876398503444,
      0.010438227794284938,
      -0.03850009560009138
    ],
    "ci95_pp": [
      -0.0782121599612523,
      -0.001796353054309141
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__cutoff_point_minus_dimensionless_point

```json
{
  "all": {
    "mean_gain_difference_pp": -2.217091540729085,
    "scene_differences_pp": [
      -0.7836610327186961,
      -5.299334526811261,
      -1.0787387688464212,
      -1.7066318345399623
    ],
    "ci95_pp": [
      -4.244185587320051,
      -0.9311999007825587
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -2.9324015159681247,
    "scene_differences_pp": [
      -0.6713373286222346,
      -6.647302071942562,
      -1.4017154298417323,
      -3.0092512334659705
    ],
    "ci95_pp": [
      -5.335905411417355,
      -1.0365263792319834
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 2.1445962120329924,
    "scene_differences_pp": [
      0.6940284009529503,
      3.8672396467772097,
      2.7938562364357433,
      1.2232605639660665
    ],
    "ci95_pp": [
      0.9586444824595084,
      3.3305479416064765
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__cutoff_point_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -0.06795684257246137,
    "scene_differences_pp": [
      0.1852195989652916,
      -0.10758695819441666,
      -0.0871365796304091,
      -0.26232343143031134
    ],
    "ci95_pp": [
      -0.21852671848033578,
      0.11201795967536454
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -0.3662787994820865,
    "scene_differences_pp": [
      -0.4294125682574368,
      -0.2666679584789833,
      -0.14767268841371362,
      -0.6213619827782124
    ],
    "ci95_pp": [
      -0.5326884767034051,
      -0.20717032344634845
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.1086250713345318,
    "scene_differences_pp": [
      1.8039265093964802,
      0.9336598543985608,
      0.6404070003627305,
      1.0565069211803557
    ],
    "ci95_pp": [
      0.7444319805671368,
      1.5863598456470005
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__cutoff_population_minus_native_population

```json
{
  "all": {
    "mean_gain_difference_pp": 0.03938824312157141,
    "scene_differences_pp": [
      0.008064113277517215,
      0.0925708119193458,
      0.08941687124877085,
      -0.03249882395934822
    ],
    "ci95_pp": [
      -0.012217355340915503,
      0.09099384158405832
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.024624096529016204,
    "scene_differences_pp": [
      -0.006071391745732235,
      0.10791562378426045,
      0.08252275656358554,
      -0.08587060248604894
    ],
    "ci95_pp": [
      -0.04597099711589059,
      0.095219190173923
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.0602334014726813,
    "scene_differences_pp": [
      -0.0408753881390278,
      -0.30829750791547816,
      0.04229693949037161,
      0.06594235067340914
    ],
    "ci95_pp": [
      -0.22064889606401572,
      0.05411964508189038
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__cutoff_population_minus_dimensionless_population

```json
{
  "all": {
    "mean_gain_difference_pp": -2.6357234749859155,
    "scene_differences_pp": [
      -0.35196428748496267,
      -6.898185208067797,
      -1.189558824505632,
      -2.10318557988527
    ],
    "ci95_pp": [
      -5.471028612177256,
      -0.7707615559952974
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -3.742870660625882,
    "scene_differences_pp": [
      -0.4856845832331702,
      -8.789362787371534,
      -1.8200219915870175,
      -3.876413280311808
    ],
    "ci95_pp": [
      -7.047027588425405,
      -1.1528532874100939
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 3.8852790585128,
    "scene_differences_pp": [
      1.2645201162325925,
      6.724302242156233,
      5.441703922147434,
      2.110589953514941
    ],
    "ci95_pp": [
      1.6875550348737667,
      6.0830030821518335
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__cutoff_population_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 1.7277026543393301,
    "scene_differences_pp": [
      3.823880509901978,
      0.6999408123216089,
      0.47559898960753966,
      1.911390305526195
    ],
    "ci95_pp": [
      0.5877699009645743,
      3.0428955855068858
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 1.5260934335470848,
    "scene_differences_pp": [
      3.891009721471972,
      0.45615396686006227,
      0.20262261141288374,
      1.5545874344434218
    ],
    "ci95_pp": [
      0.329388289136473,
      3.0322957828189945
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.1940856430893514,
    "scene_differences_pp": [
      -2.3689140631941497,
      -0.8159221172473474,
      -0.5368773692562168,
      -1.0546290226596922
    ],
    "ci95_pp": [
      -1.980666076707449,
      -0.6663152826070856
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__cutoff_selected_minus_native_selected

```json
{
  "all": {
    "mean_gain_difference_pp": 0.03456316887695832,
    "scene_differences_pp": [
      0.019188238347800812,
      0.0707014166653197,
      0.086445431307558,
      -0.03808241081284525
    ],
    "ci95_pp": [
      -0.010886453943304009,
      0.07857342398643885
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.015461817753237295,
    "scene_differences_pp": [
      -0.007259106412604854,
      0.060093483623879784,
      0.06514454192593222,
      -0.05613164812425797
    ],
    "ci95_pp": [
      -0.03169537726843141,
      0.062619012774906
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.032770030076051504,
    "scene_differences_pp": [
      -0.026743279957964106,
      -0.09027235280865131,
      0.02012903664854182,
      -0.03419352418613242
    ],
    "ci95_pp": [
      -0.07439008459597951,
      0.006548396439873261
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__cutoff_selected_minus_dimensionless_selected

```json
{
  "all": {
    "mean_gain_difference_pp": -2.1774679034584574,
    "scene_differences_pp": [
      -0.661868884814675,
      -5.306833249813237,
      -1.0509732748573808,
      -1.6901962043485375
    ],
    "ci95_pp": [
      -4.242868256074273,
      -0.8564210798360279
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -2.945910852378072,
    "scene_differences_pp": [
      -0.6863907370859401,
      -6.667136745207291,
      -1.400128148756563,
      -3.029987778462495
    ],
    "ci95_pp": [
      -5.350384596094608,
      -1.0432594429212516
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 2.1296945435588466,
    "scene_differences_pp": [
      0.6577085136191378,
      3.7991794708607918,
      2.787443570731707,
      1.2744466190237502
    ],
    "ci95_pp": [
      0.966077566321444,
      3.29331152079625
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__cutoff_selected_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 0.014355414210373674,
    "scene_differences_pp": [
      0.43470191498360444,
      -0.08672219589933583,
      -0.06157144833859318,
      -0.22898661390418074
    ],
    "ci95_pp": [
      -0.18713282251278385,
      0.3043458872628694
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -0.3503329638299052,
    "scene_differences_pp": [
      -0.3730754751642751,
      -0.26009968324265387,
      -0.15017219634793078,
      -0.6179845005647611
    ],
    "ci95_pp": [
      -0.5285132962342343,
      -0.20513593979529232
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.090728929771001,
    "scene_differences_pp": [
      1.755353895570022,
      0.8721378836859861,
      0.6398833672730309,
      1.095540572554965
    ],
    "ci95_pp": [
      0.7537976685935144,
      1.534549892599013
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

## Diagnostics

```json
{
  "solver": {
    "instances": 376776,
    "optimal_unverified": 5,
    "risk_violations": 0
  },
  "unit_predictions_exact": true,
  "unit_probes": [
    {
      "view": "coupa_seed17",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed17",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed17",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed17",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed17",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed17",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed29",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed29",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed29",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed29",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed29",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed29",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed43",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed43",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed43",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed43",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed43",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "coupa_seed43",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed17",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed17",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed17",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed17",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed17",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed17",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed29",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed29",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed29",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed29",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed29",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed29",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed43",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed43",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed43",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed43",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed43",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "deathCircle_seed43",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed17",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed17",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed17",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed17",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed17",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed17",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed29",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed29",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed29",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed29",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed29",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed29",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed43",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed43",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed43",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed43",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed43",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "gates_seed43",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed17",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed17",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed17",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed17",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed17",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed17",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed29",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed29",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed29",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed29",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed29",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed29",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed43",
      "action": "damped_velocity_005",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed43",
      "action": "damped_velocity_005",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed43",
      "action": "transformer",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed43",
      "action": "transformer",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed43",
      "action": "eqmotion",
      "factor": 0.01,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    },
    {
      "view": "hyang_seed43",
      "action": "eqmotion",
      "factor": 100.0,
      "feature_changed_rows": 0,
      "prediction_changed_rows": 0,
      "max_feature_difference": 0.0,
      "max_prediction_difference": 0.0
    }
  ]
}
```
