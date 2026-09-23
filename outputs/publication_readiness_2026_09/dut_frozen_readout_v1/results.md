# DUT Frozen-Chain Readout

## Material Passport

Fresh frozen-model inference on the full registered DUT population; existing source-only
weights and source audits were hash-verified. No new fitting, threshold selection,
independent confirmation, safety certificate or deployment. DroneCrowd stays closed.

27 recordings / 2 physical sites; 9,147 query frames, 420,364 past-eligible target windows,
399,729 complete future paths; 20,635 incomplete/unknown paths retained at inference.
426,337 visible-agent instances and 855 unknown-CV context instances.
Counts are unique query/agent instances before replication across12 model views, not IID samples.

Eight observed / twelve predicted native annotation frames, stride1. Source models used
SDD stride12: these are not matched seconds. Coordinates remain dataset-local unverified.

This newly registered descriptive readout uses complete-path ADE; the prior SDD
manuscript also reports available-point ADE. Do not pool them or compare their headline
percentages as an improvement. Here improvement is the ratio of equal-site mean errors;
mean site-relative gain is separately exported, not silently substituted.

## Every Fixed View

Native complete-path ADE, site-equal aggregation. Improvement is relative to causal CV.
Easy degradation is the maximum of the two site-specific relative ADE changes.
Intervals resample only two sites: descriptive, coarse, and not population-risk evidence.

| View | Arm | ADE | Improvement % | Easy worst-site degradation % | Hard improvement % | Two-site interval % |
|---|---|---:|---:|---:|---:|---|
| transformer_seed17_bounded_fraction | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed17_bounded_fraction | uncontrolled | 3.1280 | 18.6123 | 67.6536 | 18.9252 | [18.60598912865613, 18.61826710872574] |
| transformer_seed17_bounded_fraction | strict | 3.7847 | 1.5279 | 0.0000 | 1.9483 | [0.7421382991418267, 2.368600081340642] |
| transformer_seed17_bounded_fraction | full_independent | 3.7847 | 1.5279 | 0.0000 | 1.9483 | [0.7421382991418267, 2.368600081340642] |
| transformer_seed17_bounded_fraction | full_unary | 3.7847 | 1.5279 | 0.0000 | 1.9483 | [0.7421382991418267, 2.368600081340642] |
| transformer_seed17_bounded_fraction | full_joint | 3.7847 | 1.5279 | 0.0000 | 1.9483 | [0.7421382991418267, 2.368600081340642] |
| transformer_seed17_bounded_fraction | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed17_bounded_fraction | half_independent | 3.8125 | 0.8038 | 0.0000 | 1.0266 | [0.3451414148858063, 1.2945510549823764] |
| transformer_seed17_bounded_fraction | half_unary | 3.8125 | 0.8034 | 0.0000 | 1.0261 | [0.3448722609877193, 1.2939652833089457] |
| transformer_seed17_bounded_fraction | half_joint | 3.8125 | 0.8035 | 0.0000 | 1.0263 | [0.3448722609877193, 1.294209864187166] |
| transformer_seed17_bounded_fraction | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed17_matched_fraction_forest | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed17_matched_fraction_forest | uncontrolled | 3.1280 | 18.6123 | 67.6536 | 18.9252 | [18.60598912865613, 18.61826710872574] |
| transformer_seed17_matched_fraction_forest | strict | 3.7975 | 1.1932 | 0.0000 | 1.5204 | [0.5644995637071384, 1.8657830216248599] |
| transformer_seed17_matched_fraction_forest | full_independent | 3.7975 | 1.1932 | 0.0000 | 1.5204 | [0.5644995637071384, 1.8657830216248599] |
| transformer_seed17_matched_fraction_forest | full_unary | 3.7975 | 1.1932 | 0.0000 | 1.5204 | [0.5644995637071384, 1.8657830216248599] |
| transformer_seed17_matched_fraction_forest | full_joint | 3.7975 | 1.1932 | 0.0000 | 1.5204 | [0.5644995637071384, 1.8657830216248599] |
| transformer_seed17_matched_fraction_forest | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed17_matched_fraction_forest | half_independent | 3.8236 | 0.5140 | 0.0000 | 0.6550 | [0.22708390126700922, 0.821010069567507] |
| transformer_seed17_matched_fraction_forest | half_unary | 3.8236 | 0.5137 | 0.0000 | 0.6546 | [0.2267126820679083, 0.8206997015723718] |
| transformer_seed17_matched_fraction_forest | half_joint | 3.8236 | 0.5137 | 0.0000 | 0.6546 | [0.2267126820679083, 0.8206997015723718] |
| transformer_seed17_matched_fraction_forest | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed29_bounded_fraction | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed29_bounded_fraction | uncontrolled | 3.1148 | 18.9571 | 69.1918 | 19.4190 | [18.923060795750335, 18.993597876375993] |
| transformer_seed29_bounded_fraction | strict | 3.7804 | 1.6381 | 0.0000 | 2.0904 | [0.8989888834620839, 2.42894093936357] |
| transformer_seed29_bounded_fraction | full_independent | 3.7804 | 1.6381 | 0.0000 | 2.0904 | [0.8989888834620839, 2.42894093936357] |
| transformer_seed29_bounded_fraction | full_unary | 3.7804 | 1.6381 | 0.0000 | 2.0904 | [0.8989888834620839, 2.42894093936357] |
| transformer_seed29_bounded_fraction | full_joint | 3.7804 | 1.6381 | 0.0000 | 2.0904 | [0.8989888834620839, 2.42894093936357] |
| transformer_seed29_bounded_fraction | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed29_bounded_fraction | half_independent | 3.8099 | 0.8720 | 0.0000 | 1.1149 | [0.45938217625198857, 1.3135225416884289] |
| transformer_seed29_bounded_fraction | half_unary | 3.8099 | 0.8717 | 0.0000 | 1.1144 | [0.459390084379557, 1.312772657598089] |
| transformer_seed29_bounded_fraction | half_joint | 3.8099 | 0.8717 | 0.0000 | 1.1144 | [0.45905526528858237, 1.313112459866723] |
| transformer_seed29_bounded_fraction | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed29_matched_fraction_forest | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed29_matched_fraction_forest | uncontrolled | 3.1148 | 18.9571 | 69.1918 | 19.4190 | [18.923060795750335, 18.993597876375993] |
| transformer_seed29_matched_fraction_forest | strict | 3.7967 | 1.2150 | 0.0000 | 1.5474 | [0.5991031172048893, 1.8739720486642115] |
| transformer_seed29_matched_fraction_forest | full_independent | 3.7967 | 1.2150 | 0.0000 | 1.5474 | [0.5991031172048893, 1.8739720486642115] |
| transformer_seed29_matched_fraction_forest | full_unary | 3.7967 | 1.2150 | 0.0000 | 1.5474 | [0.5991031172048893, 1.8739720486642115] |
| transformer_seed29_matched_fraction_forest | full_joint | 3.7967 | 1.2150 | 0.0000 | 1.5474 | [0.5991031172048893, 1.8739720486642115] |
| transformer_seed29_matched_fraction_forest | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed29_matched_fraction_forest | half_independent | 3.8230 | 0.5312 | 0.0000 | 0.6761 | [0.2474076746913515, 0.8349303425501489] |
| transformer_seed29_matched_fraction_forest | half_unary | 3.8230 | 0.5310 | 0.0000 | 0.6757 | [0.24712808709184514, 0.8346534698665606] |
| transformer_seed29_matched_fraction_forest | half_joint | 3.8230 | 0.5310 | 0.0000 | 0.6757 | [0.24712808709184514, 0.8346534698665606] |
| transformer_seed29_matched_fraction_forest | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed43_bounded_fraction | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed43_bounded_fraction | uncontrolled | 3.3048 | 14.0140 | 78.6076 | 14.0653 | [13.808817119159185, 14.205862023736014] |
| transformer_seed43_bounded_fraction | strict | 3.7817 | 1.6061 | 0.0000 | 2.0392 | [1.0180704276518153, 2.2352736460251355] |
| transformer_seed43_bounded_fraction | full_independent | 3.7817 | 1.6061 | 0.0000 | 2.0392 | [1.0180704276518153, 2.2352736460251355] |
| transformer_seed43_bounded_fraction | full_unary | 3.7817 | 1.6061 | 0.0000 | 2.0392 | [1.0180704276518153, 2.2352736460251355] |
| transformer_seed43_bounded_fraction | full_joint | 3.7817 | 1.6061 | 0.0000 | 2.0392 | [1.0180704276518153, 2.2352736460251355] |
| transformer_seed43_bounded_fraction | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed43_bounded_fraction | half_independent | 3.8070 | 0.9476 | 0.0000 | 1.2032 | [0.6104853994547694, 1.308224267643153] |
| transformer_seed43_bounded_fraction | half_unary | 3.8070 | 0.9471 | 0.0000 | 1.2026 | [0.6095692695374586, 1.3081619912115576] |
| transformer_seed43_bounded_fraction | half_joint | 3.8070 | 0.9470 | 0.0000 | 1.2025 | [0.6093491662732553, 1.308224267643153] |
| transformer_seed43_bounded_fraction | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed43_matched_fraction_forest | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed43_matched_fraction_forest | uncontrolled | 3.3048 | 14.0140 | 78.6076 | 14.0653 | [13.808817119159185, 14.205862023736014] |
| transformer_seed43_matched_fraction_forest | strict | 3.8107 | 0.8507 | 0.0000 | 1.0847 | [0.3460020714059406, 1.3907479337070656] |
| transformer_seed43_matched_fraction_forest | full_independent | 3.8107 | 0.8507 | 0.0000 | 1.0847 | [0.3460020714059406, 1.3907479337070656] |
| transformer_seed43_matched_fraction_forest | full_unary | 3.8107 | 0.8507 | 0.0000 | 1.0847 | [0.3460020714059406, 1.3907479337070656] |
| transformer_seed43_matched_fraction_forest | full_joint | 3.8107 | 0.8507 | 0.0000 | 1.0847 | [0.3460020714059406, 1.3907479337070656] |
| transformer_seed43_matched_fraction_forest | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| transformer_seed43_matched_fraction_forest | half_independent | 3.8293 | 0.3653 | 0.0000 | 0.4655 | [0.13812434790924602, 0.6083359514162726] |
| transformer_seed43_matched_fraction_forest | half_unary | 3.8293 | 0.3652 | 0.0000 | 0.4654 | [0.13807726424456745, 0.6082233992258901] |
| transformer_seed43_matched_fraction_forest | half_joint | 3.8293 | 0.3652 | 0.0000 | 0.4654 | [0.13807726424456745, 0.6082529780622419] |
| transformer_seed43_matched_fraction_forest | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed17_bounded_fraction | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed17_bounded_fraction | uncontrolled | 2.3000 | 40.1566 | 79.0330 | 43.8116 | [38.47110229037106, 41.73187427460647] |
| eqmotion_seed17_bounded_fraction | strict | 3.7943 | 1.2760 | 0.0000 | 1.6134 | [1.143286857678253, 1.4180688424502823] |
| eqmotion_seed17_bounded_fraction | full_independent | 3.7943 | 1.2760 | 0.0000 | 1.6134 | [1.143286857678253, 1.4180688424502823] |
| eqmotion_seed17_bounded_fraction | full_unary | 3.7943 | 1.2760 | 0.0000 | 1.6134 | [1.143286857678253, 1.4180688424502823] |
| eqmotion_seed17_bounded_fraction | full_joint | 3.7943 | 1.2760 | 0.0000 | 1.6134 | [1.143286857678253, 1.4180688424502823] |
| eqmotion_seed17_bounded_fraction | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed17_bounded_fraction | half_independent | 3.8191 | 0.6326 | 0.0000 | 0.8038 | [0.515660084576385, 0.7576666337599139] |
| eqmotion_seed17_bounded_fraction | half_unary | 3.8191 | 0.6307 | 0.0000 | 0.8014 | [0.5141566824952114, 0.7553560642217579] |
| eqmotion_seed17_bounded_fraction | half_joint | 3.8191 | 0.6307 | 0.0000 | 0.8015 | [0.5142881443073191, 0.7553560642217579] |
| eqmotion_seed17_bounded_fraction | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed17_matched_fraction_forest | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed17_matched_fraction_forest | uncontrolled | 2.3000 | 40.1566 | 79.0330 | 43.8116 | [38.47110229037106, 41.73187427460647] |
| eqmotion_seed17_matched_fraction_forest | strict | 3.8131 | 0.7889 | 0.0000 | 1.0018 | [0.6057880314307221, 0.9848102504226042] |
| eqmotion_seed17_matched_fraction_forest | full_independent | 3.8131 | 0.7889 | 0.0000 | 1.0018 | [0.6057880314307221, 0.9848102504226042] |
| eqmotion_seed17_matched_fraction_forest | full_unary | 3.8131 | 0.7889 | 0.0000 | 1.0018 | [0.6057880314307221, 0.9848102504226042] |
| eqmotion_seed17_matched_fraction_forest | full_joint | 3.8131 | 0.7889 | 0.0000 | 1.0018 | [0.6057880314307221, 0.9848102504226042] |
| eqmotion_seed17_matched_fraction_forest | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed17_matched_fraction_forest | half_independent | 3.8342 | 0.2399 | 0.0000 | 0.3037 | [0.20503434174391033, 0.2772590590211443] |
| eqmotion_seed17_matched_fraction_forest | half_unary | 3.8342 | 0.2396 | 0.0000 | 0.3033 | [0.20503434174391033, 0.2765719201986602] |
| eqmotion_seed17_matched_fraction_forest | half_joint | 3.8342 | 0.2396 | 0.0000 | 0.3033 | [0.20503434174391033, 0.2765719201986602] |
| eqmotion_seed17_matched_fraction_forest | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed29_bounded_fraction | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed29_bounded_fraction | uncontrolled | 2.2384 | 41.7595 | 82.0183 | 45.5492 | [39.60413962283884, 43.773926255906986] |
| eqmotion_seed29_bounded_fraction | strict | 3.7940 | 1.2859 | 0.0000 | 1.6270 | [1.1819759060942558, 1.3830188752576162] |
| eqmotion_seed29_bounded_fraction | full_independent | 3.7940 | 1.2859 | 0.0000 | 1.6270 | [1.1819759060942558, 1.3830188752576162] |
| eqmotion_seed29_bounded_fraction | full_unary | 3.7940 | 1.2859 | 0.0000 | 1.6270 | [1.1819759060942558, 1.3830188752576162] |
| eqmotion_seed29_bounded_fraction | full_joint | 3.7940 | 1.2859 | 0.0000 | 1.6270 | [1.1819759060942558, 1.3830188752576162] |
| eqmotion_seed29_bounded_fraction | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed29_bounded_fraction | half_independent | 3.8193 | 0.6267 | 0.0000 | 0.7956 | [0.6075853589061662, 0.6446511294284327] |
| eqmotion_seed29_bounded_fraction | half_unary | 3.8193 | 0.6265 | 0.0000 | 0.7953 | [0.6066024970730118, 0.6450763794890816] |
| eqmotion_seed29_bounded_fraction | half_joint | 3.8193 | 0.6262 | 0.0000 | 0.7950 | [0.6066024970730118, 0.6445999279519329] |
| eqmotion_seed29_bounded_fraction | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed29_matched_fraction_forest | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed29_matched_fraction_forest | uncontrolled | 2.2384 | 41.7595 | 82.0183 | 45.5492 | [39.60413962283884, 43.773926255906986] |
| eqmotion_seed29_matched_fraction_forest | strict | 3.8183 | 0.6523 | 0.0000 | 0.8276 | [0.5010437221395844, 0.8141864430343168] |
| eqmotion_seed29_matched_fraction_forest | full_independent | 3.8183 | 0.6523 | 0.0000 | 0.8276 | [0.5010437221395844, 0.8141864430343168] |
| eqmotion_seed29_matched_fraction_forest | full_unary | 3.8183 | 0.6523 | 0.0000 | 0.8276 | [0.5010437221395844, 0.8141864430343168] |
| eqmotion_seed29_matched_fraction_forest | full_joint | 3.8183 | 0.6523 | 0.0000 | 0.8276 | [0.5010437221395844, 0.8141864430343168] |
| eqmotion_seed29_matched_fraction_forest | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed29_matched_fraction_forest | half_independent | 3.8367 | 0.1742 | 0.0000 | 0.2203 | [0.15478451844386643, 0.19506513227191388] |
| eqmotion_seed29_matched_fraction_forest | half_unary | 3.8367 | 0.1740 | 0.0000 | 0.2201 | [0.15478451844386643, 0.19462618503063814] |
| eqmotion_seed29_matched_fraction_forest | half_joint | 3.8367 | 0.1740 | 0.0000 | 0.2201 | [0.15478451844386643, 0.19462618503063814] |
| eqmotion_seed29_matched_fraction_forest | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed43_bounded_fraction | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed43_bounded_fraction | uncontrolled | 2.2128 | 42.4244 | 72.7390 | 46.1052 | [40.503036150442405, 44.22026713304315] |
| eqmotion_seed43_bounded_fraction | strict | 3.7449 | 2.5614 | 0.0000 | 3.2414 | [2.328182620580623, 2.779362244805992] |
| eqmotion_seed43_bounded_fraction | full_independent | 3.7449 | 2.5614 | 0.0000 | 3.2414 | [2.328182620580623, 2.779362244805992] |
| eqmotion_seed43_bounded_fraction | full_unary | 3.7449 | 2.5614 | 0.0000 | 3.2414 | [2.328182620580623, 2.779362244805992] |
| eqmotion_seed43_bounded_fraction | full_joint | 3.7449 | 2.5614 | 0.0000 | 3.2414 | [2.328182620580623, 2.779362244805992] |
| eqmotion_seed43_bounded_fraction | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed43_bounded_fraction | half_independent | 3.7867 | 1.4744 | 0.0000 | 1.8704 | [1.4512620788716473, 1.4959942153512054] |
| eqmotion_seed43_bounded_fraction | half_unary | 3.7868 | 1.4728 | 0.0000 | 1.8684 | [1.4500214810792191, 1.4941296843700722] |
| eqmotion_seed43_bounded_fraction | half_joint | 3.7868 | 1.4727 | 0.0000 | 1.8682 | [1.449006529087756, 1.4947965979496733] |
| eqmotion_seed43_bounded_fraction | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed43_matched_fraction_forest | floor | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed43_matched_fraction_forest | uncontrolled | 2.2128 | 42.4244 | 72.7390 | 46.1052 | [40.503036150442405, 44.22026713304315] |
| eqmotion_seed43_matched_fraction_forest | strict | 3.8158 | 0.7186 | 0.0000 | 0.9126 | [0.5656905682020889, 0.8821478058184318] |
| eqmotion_seed43_matched_fraction_forest | full_independent | 3.8158 | 0.7186 | 0.0000 | 0.9126 | [0.5656905682020889, 0.8821478058184318] |
| eqmotion_seed43_matched_fraction_forest | full_unary | 3.8158 | 0.7186 | 0.0000 | 0.9126 | [0.5656905682020889, 0.8821478058184318] |
| eqmotion_seed43_matched_fraction_forest | full_joint | 3.8158 | 0.7186 | 0.0000 | 0.9126 | [0.5656905682020889, 0.8821478058184318] |
| eqmotion_seed43_matched_fraction_forest | full_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |
| eqmotion_seed43_matched_fraction_forest | half_independent | 3.8355 | 0.2047 | 0.0000 | 0.2597 | [0.18859634001777642, 0.22192637736906753] |
| eqmotion_seed43_matched_fraction_forest | half_unary | 3.8355 | 0.2044 | 0.0000 | 0.2593 | [0.18859634001777642, 0.22125286819178083] |
| eqmotion_seed43_matched_fraction_forest | half_joint | 3.8355 | 0.2044 | 0.0000 | 0.2593 | [0.18859634001777642, 0.22125286819178083] |
| eqmotion_seed43_matched_fraction_forest | half_scene_uniform | 3.8434 | -0.0000 | 0.0000 | -0.0000 | [-0.0, 0.0] |

## Fixed Baseline Controls

No strongest-DUT winner is selected for later testing.

| Baseline | Native ADE | Native FDE | Improvement over CV % |
|---|---:|---:|---:|
| constant_position | 9.1213 | 16.7846 | -137.3246 |
| constant_velocity_causal_fd | 3.8434 | 6.8861 | 0.0000 |
| damped_velocity_005 | 3.6348 | 6.6285 | 5.4280 |
| damped_velocity_010 | 4.0158 | 8.0431 | -4.4860 |
| damped_velocity_020 | 5.0618 | 10.7853 | -31.7009 |
| constant_acceleration_causal | 27.2149 | 69.5907 | -608.0977 |
| constant_turn_rate | 8.3310 | 15.8870 | -116.7619 |

## Interpretation Boundary

Per-site/subset metrics, full-population missing-label gain bounds, rejected outputs and
unmatched solver counts are in metrics.json. Complete-case scores do not identify the
unknown absolute error of missing future trajectories. Report their support and bounds
alongside the main table. Do not use a zero-switch empirical arm as a learned safety proof.

No best seed/head/policy is selected here. Two sites cannot establish a tight2% population
risk guarantee. The source-excluded fitted chain is verified, but universal historical
exposure and cross-dataset exchangeability are not certified. Offline annotation prefixes
are not sensor-time observations. Tail/physical-validity and final independent confirmation
remain separate evidence requirements. Stage5C and SMC remain off.
