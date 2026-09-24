# All Registered Results

Design-exposed SDD source-only evidence; no model promotion.
Obs8/pred12 stride12 annotation pixels. Gain is over CV, not seconds/metric prediction.

| Action | Policy | ADE gain % | Hard gain % | Worst easy degradation % (site/seed) | Zero-CV harms (row/seed) | Mean selected |
|---|---|---:|---:|---:|---:|---:|
| damped_velocity_005 | floor | 0.000000 | 0.000000 | -0.000000 | 0 | 0.0 |
| damped_velocity_005 | uncontrolled | -6.621048 | 8.003347 | 134.469180 | 21 | 175756.0 |
| damped_velocity_005 | old_strict | 3.634683 | 4.723212 | 2.494391 | 0 | 21533.7 |
| damped_velocity_005 | native_point | 0.626177 | 0.076972 | -0.617962 | 0 | 10905.7 |
| damped_velocity_005 | native_population | 1.401053 | 1.326697 | 0.115543 | 0 | 15602.3 |
| damped_velocity_005 | native_selected | 0.646430 | 0.094897 | -0.580861 | 0 | 11209.3 |
| damped_velocity_005 | dimensionless_point | 1.343867 | 1.082791 | -0.083251 | 0 | 13663.0 |
| damped_velocity_005 | dimensionless_population | 2.011759 | 2.132084 | 1.264676 | 1 | 18831.3 |
| damped_velocity_005 | dimensionless_selected | 1.359893 | 1.102011 | -0.012390 | 0 | 13908.3 |
| damped_velocity_005 | dimensionless_matched_native_population | 1.130634 | 1.009391 | 0.373273 | 0 | 12984.7 |
| transformer | floor | 0.000000 | 0.000000 | -0.000000 | 0 | 0.0 |
| transformer | uncontrolled | 7.633081 | 10.657509 | 32.743338 | 9000 | 175756.0 |
| transformer | old_strict | 2.436827 | 2.291813 | 1.066900 | 0 | 11732.7 |
| transformer | native_point | 1.655377 | 0.838074 | -0.797599 | 0 | 14153.0 |
| transformer | native_population | 3.574607 | 3.044722 | 1.243031 | 7 | 44410.3 |
| transformer | native_selected | 1.753023 | 0.884873 | -0.785465 | 0 | 17520.3 |
| transformer | dimensionless_point | 2.883486 | 2.724271 | 0.910340 | 0 | 17762.0 |
| transformer | dimensionless_population | 4.988086 | 5.252471 | 4.450332 | 5 | 51630.0 |
| transformer | dimensionless_selected | 2.986166 | 2.791706 | 0.943656 | 0 | 21355.0 |
| transformer | dimensionless_matched_native_population | 3.835997 | 4.007214 | 3.320932 | 5 | 37824.7 |
| eqmotion | floor | 0.000000 | 0.000000 | -0.000000 | 0 | 0.0 |
| eqmotion | uncontrolled | 11.043498 | 15.401484 | 47.391895 | 34584 | 175756.0 |
| eqmotion | old_strict | 1.609305 | 0.527132 | 0.452161 | 0 | 6901.0 |
| eqmotion | native_point | 1.505354 | 0.143881 | -0.469674 | 0 | 8629.0 |
| eqmotion | native_population | 3.297619 | 2.028601 | 1.345792 | 7 | 28395.7 |
| eqmotion | native_selected | 1.589097 | 0.161337 | -0.417987 | 0 | 9499.3 |
| eqmotion | dimensionless_point | 3.758440 | 3.093255 | 3.525947 | 0 | 11324.7 |
| eqmotion | dimensionless_population | 5.972731 | 5.796096 | 9.009714 | 6 | 35819.3 |
| eqmotion | dimensionless_selected | 3.801128 | 3.122710 | 3.529848 | 0 | 12091.3 |
| eqmotion | dimensionless_matched_native_population | 3.095356 | 2.363647 | 3.836807 | 3 | 22669.3 |

Positive-easy degradation excludes exact-zero CV errors; those harms are reported separately.
Repeated seeds/overlapping windows are not independent samples.

## Paired Contrasts

Three thousand paired bootstrap resamples of four physical sites; sites are already design-exposed.
All pre-registered contrasts are retained below. No multiple-comparison or independent-confirmation claim.

### damped_velocity_005__dimensionless_population_minus_native_population

```json
{
  "all": {
    "mean_gain_difference_pp": 0.6107059911527574,
    "scene_differences_pp": [
      -0.28404886703484733,
      1.6588155595755882,
      0.39856891097241887,
      0.6694883610978697
    ],
    "ci95_pp": [
      -0.045664560001668075,
      1.3437538974247958
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.8053862671983963,
    "scene_differences_pp": [
      -0.5145992943460409,
      2.0899358528074785,
      0.615741495697042,
      1.0304670146351058
    ],
    "ci95_pp": [
      -0.12833271710075422,
      1.7213872635298695
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.6525839527208727,
    "scene_differences_pp": [
      0.45906729913905986,
      -1.0384188504272363,
      -0.9041270855288719,
      -1.1268571740664424
    ],
    "ci95_pp": [
      -1.0826380122468393,
      0.08469576174748583
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__dimensionless_point_minus_native_point

```json
{
  "all": {
    "mean_gain_difference_pp": 0.7176905317555909,
    "scene_differences_pp": [
      0.26425697320520136,
      1.4272698417252982,
      0.47251641633262453,
      0.7067188957592396
    ],
    "ci95_pp": [
      0.36838669476891295,
      1.1885814853771297
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 1.00581814689234,
    "scene_differences_pp": [
      0.4945710229525835,
      1.7359726886367666,
      0.6646885815644321,
      1.1280402944155776
    ],
    "ci95_pp": [
      0.5796298022585078,
      1.468151661868683
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.4493444473528313,
    "scene_differences_pp": [
      -0.09322450912099267,
      -0.6259698481291376,
      -0.5581204900980907,
      -0.5200629420631042
    ],
    "ci95_pp": [
      -0.5994931216126292,
      -0.20944850436526719
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__dimensionless_selected_minus_native_selected

```json
{
  "all": {
    "mean_gain_difference_pp": 0.7134633573470273,
    "scene_differences_pp": [
      0.24426003413587916,
      1.4297983693867033,
      0.46764604922384656,
      0.71214897664168
    ],
    "ci95_pp": [
      0.35595304167986286,
      1.1892602893459892
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 1.0071138352281184,
    "scene_differences_pp": [
      0.4777004060523171,
      1.7428710849433648,
      0.6572763134266135,
      1.150607536490178
    ],
    "ci95_pp": [
      0.5674883597394653,
      1.471472392064177
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.4691821209341751,
    "scene_differences_pp": [
      -0.10524092240706828,
      -0.6889248440976381,
      -0.5502127278360192,
      -0.5323499893959749
    ],
    "ci95_pp": [
      -0.6497811304222223,
      -0.216483873764306
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__dimensionless_matched_native_population_minus_native_population

```json
{
  "all": {
    "mean_gain_difference_pp": -0.2704196272634629,
    "scene_differences_pp": [
      -0.8947852471277828,
      0.03866908369687394,
      -0.07148566908969922,
      -0.15407667653324353
    ],
    "ci95_pp": [
      -0.6889603526182619,
      -0.009517356360655427
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -0.31730642155183286,
    "scene_differences_pp": [
      -1.1711738584067821,
      0.12708602989038864,
      -0.04903114737853098,
      -0.176106710312407
    ],
    "ci95_pp": [
      -0.8906381806497193,
      0.051287844839689734
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.28123303056334925,
    "scene_differences_pp": [
      0.04147323560454996,
      -0.5595332362148042,
      -0.3676758882160236,
      -0.23919623342711915
    ],
    "ci95_pp": [
      -0.47944898551788295,
      -0.060814045350593426
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__native_point_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -3.0085061415910967,
    "scene_differences_pp": [
      -1.9101985879222294,
      -4.859350413142571,
      -1.9487126626581253,
      -3.3157629026414615
    ],
    "ci95_pp": [
      -4.13169097552146,
      -1.9294556252901773
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -4.646239736768676,
    "scene_differences_pp": [
      -3.864742119295439,
      -6.137047229145443,
      -2.882614656228255,
      -5.700554942405566
    ],
    "ci95_pp": [
      -5.918801085775504,
      -3.373678387761847
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 2.3841215810896794,
    "scene_differences_pp": [
      1.4268755179394077,
      3.1967251471315405,
      2.9563630517868145,
      1.9565226075009545
    ],
    "ci95_pp": [
      1.691699062720181,
      3.076544099459178
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__native_population_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -2.2336293792458495,
    "scene_differences_pp": [
      -0.6837856082977711,
      -4.248750806546441,
      -1.6560848697872999,
      -2.345896232351885
    ],
    "ci95_pp": [
      -3.600584322356656,
      -1.0993132643112995
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -3.3965147700726037,
    "scene_differences_pp": [
      -1.5528943707676546,
      -5.44855130159112,
      -2.483849869195631,
      -4.1007635387360075
    ],
    "ci95_pp": [
      -4.774657420163564,
      -2.0183721199816427
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.8107654477936246,
    "scene_differences_pp": [
      0.442449487793517,
      2.8112126192001208,
      2.730145537751971,
      1.2592541464288898
    ],
    "ci95_pp": [
      0.8508518171112034,
      2.7706790784760456
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__native_selected_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -2.9882528384157667,
    "scene_differences_pp": [
      -1.8726974472227043,
      -4.841133411057029,
      -1.9442310052865786,
      -3.2949494900967546
    ],
    "ci95_pp": [
      -4.116907809614417,
      -1.9084642262546414
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -4.628315447399631,
    "scene_differences_pp": [
      -3.828693751823109,
      -6.124042968213861,
      -2.8793278243174236,
      -5.681197245244129
    ],
    "ci95_pp": [
      -5.902620106728995,
      -3.354010788070266
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 2.3797783839925284,
    "scene_differences_pp": [
      1.433586155116351,
      3.2018371038215365,
      2.9582236931638417,
      1.9254665838683849
    ],
    "ci95_pp": [
      1.6795263694923679,
      3.0800303984926893
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__dimensionless_point_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -2.290815609835506,
    "scene_differences_pp": [
      -1.645941614717028,
      -3.4320805714172726,
      -1.4761962463255007,
      -2.609044006882222
    ],
    "ci95_pp": [
      -3.0205622891497477,
      -1.5610689305212642
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -3.6404215898763357,
    "scene_differences_pp": [
      -3.3701710963428555,
      -4.401074540508676,
      -2.217926074663823,
      -4.572514647989989
    ],
    "ci95_pp": [
      -4.486794594249332,
      -2.763713191125036
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.9347771337368478,
    "scene_differences_pp": [
      1.333651008818415,
      2.570755299002403,
      2.3982425616887237,
      1.4364596654378503
    ],
    "ci95_pp": [
      1.3850553371281324,
      2.4844989303455636
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__dimensionless_population_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -1.622923388093092,
    "scene_differences_pp": [
      -0.9678344753326185,
      -2.589935246970853,
      -1.257515958814881,
      -1.6764078712540154
    ],
    "ci95_pp": [
      -2.2568304249318603,
      -1.1126752170737497
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -2.5911285028742075,
    "scene_differences_pp": [
      -2.0674936651136955,
      -3.358615448783642,
      -1.8681083734985893,
      -3.0702965241009013
    ],
    "ci95_pp": [
      -3.214455986442272,
      -1.9678010193061424
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.158181495072752,
    "scene_differences_pp": [
      0.9015167869325769,
      1.7727937687728845,
      1.8260184522230993,
      0.1323969723624474
    ],
    "ci95_pp": [
      0.5169568796475121,
      1.799406110497992
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### damped_velocity_005__dimensionless_selected_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -2.2747894810687397,
    "scene_differences_pp": [
      -1.628437413086825,
      -3.4113350416703265,
      -1.476584956062732,
      -2.5828005134550747
    ],
    "ci95_pp": [
      -2.997067777562701,
      -1.5525111845747785
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -3.6212016121715127,
    "scene_differences_pp": [
      -3.350993345770792,
      -4.381171883270497,
      -2.22205151089081,
      -4.530589708753951
    ],
    "ci95_pp": [
      -4.455880796012224,
      -2.7618316039857316
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.9105962630583533,
    "scene_differences_pp": [
      1.3283452327092826,
      2.5129122597238984,
      2.4080109653278226,
      1.39311659447241
    ],
    "ci95_pp": [
      1.3607309135908463,
      2.4604616125258607
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__dimensionless_population_minus_native_population

```json
{
  "all": {
    "mean_gain_difference_pp": 1.413479301914278,
    "scene_differences_pp": [
      0.17322302600118,
      1.445629883959909,
      1.767596145765049,
      2.2674681519309736
    ],
    "ci95_pp": [
      0.5718163059421473,
      2.0620085849382077
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 2.2077486798471937,
    "scene_differences_pp": [
      0.1758247451120969,
      1.754733754250637,
      2.6624727567109097,
      4.237963463315131
    ],
    "ci95_pp": [
      0.7974867480118001,
      3.6171560360490074
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.7309608256813158,
    "scene_differences_pp": [
      -0.19797459136026152,
      -0.5544239713251131,
      -3.634235537516617,
      -2.5372092025232718
    ],
    "ci95_pp": [
      -3.085722370019944,
      -0.3761992813426873
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__dimensionless_point_minus_native_point

```json
{
  "all": {
    "mean_gain_difference_pp": 1.228108630393332,
    "scene_differences_pp": [
      0.47336534638215966,
      1.1955174551921521,
      1.4502902265442708,
      1.7932614934547453
    ],
    "ci95_pp": [
      0.7175965664226874,
      1.6438254838890969
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 1.8861963763309286,
    "scene_differences_pp": [
      0.6318847023496943,
      1.486793152993171,
      2.057253502491163,
      3.3688541474896865
    ],
    "ci95_pp": [
      0.9882269023850615,
      2.898338898865558
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.1359275276599328,
    "scene_differences_pp": [
      -0.45001293695748856,
      -0.47768317596562637,
      -2.116602403280843,
      -1.4994115944357733
    ],
    "ci95_pp": [
      -1.8080069988583083,
      -0.46384805646155747
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__dimensionless_selected_minus_native_selected

```json
{
  "all": {
    "mean_gain_difference_pp": 1.2331429737884312,
    "scene_differences_pp": [
      0.475492285146617,
      1.2125614343031166,
      1.4400587857658298,
      1.8044593899381618
    ],
    "ci95_pp": [
      0.7166339103014202,
      1.6564849010294003
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 1.9068331947508481,
    "scene_differences_pp": [
      0.6499838505526268,
      1.50777175390564,
      2.062904906948193,
      3.406672267596933
    ],
    "ci95_pp": [
      1.0032141146515183,
      2.93194713917411
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.1435168249747858,
    "scene_differences_pp": [
      -0.4736496906074361,
      -0.4492753628394519,
      -2.1831831512123756,
      -1.467959095239879
    ],
    "ci95_pp": [
      -1.8255711232261274,
      -0.461462526723444
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__dimensionless_matched_native_population_minus_native_population

```json
{
  "all": {
    "mean_gain_difference_pp": 0.2613900158158683,
    "scene_differences_pp": [
      -1.2069504366127521,
      0.2665286590132987,
      0.9312165732721311,
      1.0547652675907955
    ],
    "ci95_pp": [
      -0.6724086841415313,
      0.9929909204314633
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.9624922258007456,
    "scene_differences_pp": [
      -0.9061806038645925,
      0.3696802766692264,
      1.5903927062484557,
      2.7960765241498926
    ],
    "ci95_pp": [
      -0.28203727633633047,
      2.193234615199174
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.2412675440866243,
    "scene_differences_pp": [
      -0.817404018177037,
      -0.3766656061453655,
      -2.1000421115252688,
      -1.6709584404988265
    ],
    "ci95_pp": [
      -1.8855002760120478,
      -0.5970348121612012
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__native_point_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -0.7814496944897137,
    "scene_differences_pp": [
      0.24722285156472124,
      -3.005900784184956,
      -0.474229070213561,
      0.1071082248749411
    ],
    "ci95_pp": [
      -2.2276485319199817,
      0.17716553821983116
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -1.4537386962675591,
    "scene_differences_pp": [
      -1.150158565139714,
      -3.79041516571198,
      -0.7936426642295658,
      -0.08073838998897687
    ],
    "ci95_pp": [
      -3.0412220403413768,
      -0.34809343377666113
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.6487476016271763,
    "scene_differences_pp": [
      3.1543264655855547,
      1.6771130680948265,
      1.1359891959680346,
      0.6275616768602887
    ],
    "ci95_pp": [
      0.8817754364141617,
      2.6497421481811747
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__native_population_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 1.1377798712996738,
    "scene_differences_pp": [
      3.252692003318824,
      -1.577334091219218,
      0.2565311820062921,
      2.6192303910927976
    ],
    "ci95_pp": [
      -0.6604014546064629,
      2.9359611972058106
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.7529087610269881,
    "scene_differences_pp": [
      2.8164722339575676,
      -2.3050769466219534,
      -0.18732104132143101,
      2.6875607980937692
    ],
    "ci95_pp": [
      -1.2461989939716922,
      2.7520165160256687
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -0.164524331763935,
    "scene_differences_pp": [
      -0.03611093041725466,
      0.8902733799810769,
      0.020195099626474367,
      -1.5324548762460366
    ],
    "ci95_pp": [
      -1.144292382277909,
      0.658677302381494
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__native_selected_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -0.6838036208834469,
    "scene_differences_pp": [
      0.4778360772313306,
      -2.9661422817407024,
      -0.4522990093156154,
      0.20539073029119947
    ],
    "ci95_pp": [
      -2.173259028732727,
      0.34161340376126503
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -1.4069403922303323,
    "scene_differences_pp": [
      -1.053308189041413,
      -3.757147144343298,
      -0.7890160454376294,
      -0.028290190098989587
    ],
    "ci95_pp": [
      -3.0151143696168807,
      -0.28454468983459547
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.619790217646536,
    "scene_differences_pp": [
      3.0840815799807912,
      1.6792058368905405,
      1.1087484124826186,
      0.6071250412321927
    ],
    "ci95_pp": [
      0.8579367268574056,
      2.590248288106248
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__dimensionless_point_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 0.44665893590361827,
    "scene_differences_pp": [
      0.7205881979468809,
      -1.810383328992804,
      0.9760611563307098,
      1.9003697183296864
    ],
    "ci95_pp": [
      -1.1137722076619254,
      1.605424338233985
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.4324576800633696,
    "scene_differences_pp": [
      -0.5182738627900196,
      -2.303622012718809,
      1.2636108382615974,
      3.2881157575007096
    ],
    "ci95_pp": [
      -1.4118137999737073,
      2.336518352428027
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 0.5128200739672434,
    "scene_differences_pp": [
      2.704313528628066,
      1.1994298921292001,
      -0.9806132073128082,
      -0.8718499175754846
    ],
    "ci95_pp": [
      -0.9262315624441464,
      1.9518717103786334
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__dimensionless_population_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 2.5512591732139516,
    "scene_differences_pp": [
      3.425915029320004,
      -0.13170420725930887,
      2.024127327771341,
      4.886698543023771
    ],
    "ci95_pp": [
      0.7577006018855194,
      4.171055739210663
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 2.960657440874182,
    "scene_differences_pp": [
      2.9922969790696645,
      -0.5503431923713165,
      2.4751517153894786,
      6.9255242614089
    ],
    "ci95_pp": [
      0.3353168504889288,
      5.812931124904044
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.895485157445251,
    "scene_differences_pp": [
      -0.23408552177751618,
      0.3358494086559638,
      -3.6140404378901425,
      -4.069664078769309
    ],
    "ci95_pp": [
      -3.841852258329726,
      0.05088194343922381
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### transformer__dimensionless_selected_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 0.5493393529049844,
    "scene_differences_pp": [
      0.9533283623779476,
      -1.7535808474375858,
      0.9877597764502144,
      2.0098501202293613
    ],
    "ci95_pp": [
      -1.0682456914656357,
      1.7457196807665079
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.4998928025205157,
    "scene_differences_pp": [
      -0.40332433848878635,
      -2.249375390437658,
      1.2738888615105637,
      3.3783820774979434
    ],
    "ci95_pp": [
      -1.3685593274506025,
      2.432955473501261
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 0.47627339267175006,
    "scene_differences_pp": [
      2.610431889373355,
      1.2299304740510886,
      -1.074434738729757,
      -0.8608340540076864
    ],
    "ci95_pp": [
      -0.9676343963687217,
      1.920181181712222
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__dimensionless_population_minus_native_population

```json
{
  "all": {
    "mean_gain_difference_pp": 2.6751117181074866,
    "scene_differences_pp": [
      0.3600284007624799,
      6.990756019987142,
      1.278975695754403,
      2.0706867559259217
    ],
    "ci95_pp": [
      0.7876929895533403,
      5.562810938928957
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 3.767494757154899,
    "scene_differences_pp": [
      0.479613191487438,
      8.897278411155796,
      1.902544748150603,
      3.790542677825759
    ],
    "ci95_pp": [
      1.1910789698190205,
      7.148594995404498
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -3.9455124599854816,
    "scene_differences_pp": [
      -1.3053955043716203,
      -7.032599750071711,
      -5.399406982657062,
      -2.0446476028415317
    ],
    "ci95_pp": [
      -6.2160033663643866,
      -1.675021553606576
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__dimensionless_point_minus_native_point

```json
{
  "all": {
    "mean_gain_difference_pp": 2.2530861795258117,
    "scene_differences_pp": [
      0.7970420610844848,
      5.37860333758271,
      1.1633398939252193,
      1.6733594255108342
    ],
    "ci95_pp": [
      0.9801909775048521,
      4.324787476668337
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 2.949373705473662,
    "scene_differences_pp": [
      0.6640509949629836,
      6.709043511374235,
      1.4721676773633008,
      2.95223263819413
    ],
    "ci95_pp": [
      1.0681093361631422,
      5.399824552871501
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -2.181904456953179,
    "scene_differences_pp": [
      -0.7193607488428562,
      -3.963078410762244,
      -2.7834180086414584,
      -1.261760659566158
    ],
    "ci95_pp": [
      -3.373248209701851,
      -0.990560704204507
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__dimensionless_selected_minus_native_selected

```json
{
  "all": {
    "mean_gain_difference_pp": 2.212031072335416,
    "scene_differences_pp": [
      0.6810571231624758,
      5.377534666478557,
      1.1374187061649388,
      1.6521137935356922
    ],
    "ci95_pp": [
      0.9092379146637073,
      4.3175056764001525
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 2.96137267013131,
    "scene_differences_pp": [
      0.6791316306733353,
      6.727230228831171,
      1.4652726906824953,
      2.973856130338237
    ],
    "ci95_pp": [
      1.0722021606779153,
      5.411740844294002
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -2.162464573634898,
    "scene_differences_pp": [
      -0.684451793577102,
      -3.889451823669443,
      -2.767314534083165,
      -1.3086401432098826
    ],
    "ci95_pp": [
      -3.328383178876304,
      -0.9965459683934923
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__dimensionless_matched_native_population_minus_native_population

```json
{
  "all": {
    "mean_gain_difference_pp": -0.20226360199270554,
    "scene_differences_pp": [
      -1.4405410726329064,
      0.41462433689986034,
      0.07568879969844788,
      0.14117352806377603
    ],
    "ci95_pp": [
      -1.0451124224587358,
      0.3298904525995072
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 0.3350458827865582,
    "scene_differences_pp": [
      -1.0509597513682967,
      0.8564205607549447,
      0.4072625683916531,
      1.1274601533679318
    ],
    "ci95_pp": [
      -0.5741146733374863,
      0.9919403570614382
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.8354365665507688,
    "scene_differences_pp": [
      -1.3252094282031557,
      -2.3826736162321005,
      -2.571308184543386,
      -1.062555037224433
    ],
    "ci95_pp": [
      -2.4769909003877433,
      -1.1938822327137943
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__native_point_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -0.1039514813691883,
    "scene_differences_pp": [
      0.17183857059950292,
      -0.18685576896586564,
      -0.17173770470920724,
      -0.22905102240118325
    ],
    "ci95_pp": [
      -0.21472269297818924,
      0.08216498570816078
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -0.383250988987624,
    "scene_differences_pp": [
      -0.42212623459818577,
      -0.32840939791065615,
      -0.21812493593528215,
      -0.5643433875063719
    ],
    "ci95_pp": [
      -0.505359890107443,
      -0.26912526060100805
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.1459333162547185,
    "scene_differences_pp": [
      1.8292588572863862,
      1.0294986183835952,
      0.6299687725684455,
      1.095007016780447
    ],
    "ci95_pp": [
      0.7462283336214459,
      1.6293187975606884
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__native_population_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 1.688314411217759,
    "scene_differences_pp": [
      3.8158163966244607,
      0.6073700004022631,
      0.3861821183587688,
      1.9438891294855432
    ],
    "ci95_pp": [
      0.49677605938051594,
      3.0137047975689115
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 1.501469337018069,
    "scene_differences_pp": [
      3.897081113217704,
      0.3482383430758018,
      0.1200998548492982,
      1.6404580369294708
    ],
    "ci95_pp": [
      0.23416909896255,
      3.0098704206822284
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.1338522416166703,
    "scene_differences_pp": [
      -2.328038675055122,
      -0.5076246093318693,
      -0.5791743087465884,
      -1.1205713733331013
    ],
    "ci95_pp": [
      -1.8908225834779886,
      -0.5433994590392288
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__native_selected_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": -0.020207754666584643,
    "scene_differences_pp": [
      0.41551367663580363,
      -0.15742361256465554,
      -0.14801687964615118,
      -0.1909042030913355
    ],
    "ci95_pp": [
      -0.1801823722300394,
      0.27227935433568884
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": -0.3657947815831425,
    "scene_differences_pp": [
      -0.36581636875167023,
      -0.32019316686653365,
      -0.215316738273863,
      -0.5618528524405031
    ],
    "ci95_pp": [
      -0.5014379310470107,
      -0.2529416458933148
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": 1.1234989598470526,
    "scene_differences_pp": [
      1.7820971755279862,
      0.9624102364946374,
      0.619754330624489,
      1.1297340967410974
    ],
    "ci95_pp": [
      0.7472492721536411,
      1.577175440769649
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__dimensionless_point_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 2.149134698156624,
    "scene_differences_pp": [
      0.9688806316839877,
      5.191747568616845,
      0.9916021892160121,
      1.444308403109651
    ],
    "ci95_pp": [
      0.9802414104499999,
      4.141711223766636
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 2.5661227164860385,
    "scene_differences_pp": [
      0.24192476036479782,
      6.380634113463579,
      1.2540427414280186,
      2.387889250687758
    ],
    "ci95_pp": [
      0.7479837508964082,
      5.098986270454689
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.0359711406984606,
    "scene_differences_pp": [
      1.10989810844353,
      -2.933579792378649,
      -2.153449236073013,
      -0.1667536427857108
    ],
    "ci95_pp": [
      -2.543514514225831,
      0.4715722328289096
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__dimensionless_population_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 4.363426129325245,
    "scene_differences_pp": [
      4.17584479738694,
      7.598126020389405,
      1.6651578141131718,
      4.014575885411465
    ],
    "ci95_pp": [
      2.2928295599316137,
      6.70223848664492
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 5.268964094172967,
    "scene_differences_pp": [
      4.376694304705142,
      9.245516754231597,
      2.0226446029999012,
      5.43100071475523
    ],
    "ci95_pp": [
      2.8747336309387332,
      8.028311141849983
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -5.0793647016021515,
    "scene_differences_pp": [
      -3.633434179426742,
      -7.54022435940358,
      -5.978581291403651,
      -3.165218976174633
    ],
    "ci95_pp": [
      -6.759402825403615,
      -3.3993265778006876
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

### eqmotion__dimensionless_selected_minus_old_strict

```json
{
  "all": {
    "mean_gain_difference_pp": 2.191823317668831,
    "scene_differences_pp": [
      1.0965707997982794,
      5.220111053913901,
      0.9894018265187876,
      1.4612095904443567
    ],
    "ci95_pp": [
      1.0429863131585335,
      4.189225990384996
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "hard": {
    "mean_gain_difference_pp": 2.595577888548167,
    "scene_differences_pp": [
      0.31331526192166503,
      6.407037061964637,
      1.2499559524086323,
      2.412003277897734
    ],
    "ci95_pp": [
      0.7816356071651487,
      5.117766784575636
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  },
  "positive_easy": {
    "mean_gain_difference_pp": -1.0389656137878456,
    "scene_differences_pp": [
      1.0976453819508842,
      -2.9270415871748057,
      -2.147560203458676,
      -0.17890604646878527
    ],
    "ci95_pp": [
      -2.537300895316741,
      0.4593696677410495
    ],
    "resamples": 3000,
    "seed": 38113,
    "unit": "physical_scene",
    "interpretation": "conditional_development_contrast_not_independent_confirmation"
  }
}
```

## Numerical Limitations

```json
{
  "instances": 941940,
  "optimal_unverified": 12575,
  "exact_count_failures": 12569,
  "risk_violations": 0
}
```

Matched-count contrasts with failed exact-count queries are not fully equal-coverage comparisons.
Unknown-label selections remain unknown; all subset, per-scene/seed and partial-future bounds are in analysis.json.

Analysis SHA256: `122a42cdf729e52970e902847996f608d352c8ac12ad9c4d761bf78b13f35808`.
