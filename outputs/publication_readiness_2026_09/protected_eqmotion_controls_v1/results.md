# Full EqMotion Protected Motion Controls

Four design-exposed SDD sites, three seeds, observation8/prediction12 native annotation steps.
Annotation pixels; no verified metric/seconds, independent confirmation or deployment.
Twelve new forests; twelve cached-verified neural heads. No new forecasting models.

| EqMotion policy | ADE gain % [CI95] | Hard gain % | Worst site/seed easy degradation % |
|---|---:|---:|---:|
| eqmotion__uncontrolled | 11.0435 [8.224567009793134, 13.397602185227885] | 15.4015 | 47.3919 |
| eqmotion__neural__net_stop | 11.0065 [8.853700100060797, 13.159353941080205] | 13.3300 | 40.9449 |
| eqmotion__neural__strict_stop | 1.6093 [0.7162681896853484, 3.039280619820001] | 0.5271 | 0.4522 |
| eqmotion__forest__net_stop | 12.4165 [9.753080449241097, 14.537367537313845] | 15.0996 | 40.6954 |
| eqmotion__forest__strict_stop | 1.2632 [0.42882543577743293, 2.7352121987779947] | 0.0523 | 0.0000 |

## Strict and Matched-Count Contrasts

Positive difference favors EqMotion. All comparisons reported; no selected winner.

| Comparator | Head | Strict gain difference pp [CI95] | Matched difference pp [CI95] | Matched EqMotion easy degradation % | Matched comparator easy degradation % |
|---|---|---:|---:|---:|---:|
| constant_position | neural | -0.2074 [-0.36093245564601784, -0.05377004881300662] | -0.3206 [-0.5115970801490841, -0.12952474184885765] | 0.9798 | 2.1935 |
| constant_position | forest | 0.5148 [0.10142062336348512, 1.247322935855577] | 0.0699 [0.002274067131341484, 0.13760052108336973] | 0.2307 | 0.3484 |
| damped_velocity_005 | neural | -2.0254 [-3.709420242179056, -0.34556825664714963] | -1.0586 [-2.3607139803397406, 0.40757977974015536] | 0.4522 | 1.6215 |
| damped_velocity_005 | forest | -0.1512 [-1.7214033426879687, 1.4912702062088834] | -0.0837 [-1.657412440105069, 1.5932190549654779] | 0.0000 | 0.2874 |
| damped_velocity_010 | neural | -2.1354 [-4.465995954644295, -0.2343977436570721] | -1.4914 [-3.6206740950044534, 0.3443007626150074] | 0.4522 | 2.6917 |
| damped_velocity_010 | forest | -0.0131 [-1.0878637838134946, 1.146593379578062] | 0.0372 [-1.064558789656339, 1.2514070771088655] | 0.0000 | 0.3402 |
| damped_velocity_020 | neural | -2.0965 [-4.640430704513671, -0.3820504323425955] | -1.6962 [-4.098102274699578, -0.011736728374467376] | 0.4522 | 2.9923 |
| damped_velocity_020 | forest | 0.1584 [-0.12266856590691932, 0.5619715433327482] | 0.1808 [-0.11082247646856391, 0.6163199438417227] | 0.0000 | 0.0000 |
| constant_acceleration_causal | neural | 1.6107 [0.7172122194228192, 3.0424637502226957] | 0.0071 [0.001363017780159348, 0.012953591376127505] | 0.0000 | 0.0005 |
| constant_acceleration_causal | forest | 1.2632 [0.42882543577743293, 2.7352121987779947] | 0.0000 [0.0, 0.0] | 0.0000 | 0.0000 |
| constant_turn_rate | neural | 1.0458 [-0.08918856497397987, 2.80286863378828] | -0.2799 [-0.5453652782431151, -0.01446450816685596] | 0.6984 | 1.4846 |
| constant_turn_rate | forest | 1.2593 [0.4257484527656591, 2.732773360214652] | 0.0161 [0.0026825224264948044, 0.031172298336737403] | 0.0000 | 0.0041 |
| transformer | neural | -0.8275 [-2.1481569082965586, 0.4420337102535532] | -0.5221 [-1.5757272985289004, 0.5693771458669838] | 0.4522 | 1.8617 |
| transformer | forest | -0.0388 [-0.9670789917532563, 0.8015838965880928] | 0.0565 [-0.7654765440081979, 0.8742033606042199] | 0.0000 | 0.3274 |

CIs are nominal four-site development bootstraps, not multiplicity-adjusted confirmation.
Matched counts are outcome-blind offline diagnostics, not online safety certificates.
Training-only easy/hard cutoffs are shared with the prior protected-motion study.
Unknown outcomes, complete paths, zero-CV harms, tails and missing-label gain bounds remain in analysis.json.
