# Explicit Zero-Reference Atom: All Controls

Four development-exposed SDD sites, three seeds, obs8/pred12 stride12 annotation pixels.
Equal-physical-site gains over CV; not historical raw t50, strongest-baseline or independent-confirmation claims.

| Action | Policy | ADE gain % | FDE gain % | Hard gain % | Worst positive-easy degradation % | Zero-CV harms, window/seed | Mean switches |
|---|---|---:|---:|---:|---:|---:|---:|
| damped_velocity_005 | old_strict | 3.634683 | 4.649955 | 4.723212 | 2.494391 | 0 | 21533.7 |
| damped_velocity_005 | cutoff_point | 0.646366 | 0.880692 | 0.100538 | -0.591840 | 0 | 11045.3 |
| damped_velocity_005 | cutoff_population | 1.432180 | 1.795960 | 1.398032 | 0.135072 | 0 | 15715.3 |
| damped_velocity_005 | cutoff_selected | 0.667723 | 0.908936 | 0.120303 | -0.569173 | 0 | 11345.0 |
| damped_velocity_005 | atom_point | 0.600040 | 0.821704 | 0.090031 | -0.591113 | 0 | 10352.0 |
| damped_velocity_005 | atom_population | 1.228418 | 1.563855 | 1.144151 | 0.170285 | 0 | 13821.7 |
| damped_velocity_005 | atom_selected | 0.616864 | 0.844577 | 0.107342 | -0.581031 | 0 | 10587.0 |
| damped_velocity_005 | matched_point | 0.607184 | 0.830870 | 0.094630 | -0.589122 | 0 | 10352.0 |
| damped_velocity_005 | matched_population | 1.285116 | 1.625163 | 1.255722 | 0.112274 | 0 | 13821.7 |
| damped_velocity_005 | matched_selected | 0.623849 | 0.853430 | 0.112155 | -0.578004 | 0 | 10587.0 |
| transformer | old_strict | 2.436827 | 2.694480 | 2.291813 | 1.066900 | 0 | 11732.7 |
| transformer | cutoff_point | 1.673652 | 1.735010 | 0.852618 | -0.773232 | 1 | 14554.0 |
| transformer | cutoff_population | 3.604563 | 3.887128 | 3.082321 | 1.273190 | 5 | 44673.7 |
| transformer | cutoff_selected | 1.771752 | 1.845245 | 0.899853 | -0.750879 | 1 | 17919.3 |
| transformer | atom_point | 1.601488 | 1.651973 | 0.813465 | -0.682808 | 1 | 12289.7 |
| transformer | atom_population | 2.970180 | 3.146625 | 2.453534 | 1.015192 | 5 | 35645.3 |
| transformer | atom_selected | 1.683860 | 1.744065 | 0.851780 | -0.661825 | 1 | 14955.0 |
| transformer | matched_point | 1.613642 | 1.666464 | 0.824015 | -0.625786 | 1 | 12289.7 |
| transformer | matched_population | 3.233590 | 3.446282 | 2.782746 | 1.268726 | 5 | 35645.3 |
| transformer | matched_selected | 1.699651 | 1.762798 | 0.864102 | -0.605516 | 1 | 14955.0 |
| eqmotion | old_strict | 1.609305 | 1.730640 | 0.527132 | 0.452161 | 0 | 6901.0 |
| eqmotion | cutoff_point | 1.541348 | 1.656685 | 0.160853 | -0.366782 | 0 | 8734.0 |
| eqmotion | cutoff_population | 3.337008 | 3.522162 | 2.053225 | 1.458027 | 6 | 28666.7 |
| eqmotion | cutoff_selected | 1.623660 | 1.742368 | 0.176799 | -0.364199 | 0 | 9608.0 |
| eqmotion | atom_point | 1.510435 | 1.620352 | 0.157312 | -0.427184 | 0 | 8012.3 |
| eqmotion | atom_population | 2.796439 | 2.956629 | 1.511688 | 1.201478 | 8 | 24308.0 |
| eqmotion | atom_selected | 1.583270 | 1.695705 | 0.170996 | -0.428676 | 0 | 8656.7 |
| eqmotion | matched_point | 1.512458 | 1.622088 | 0.157988 | -0.382460 | 0 | 8012.3 |
| eqmotion | matched_population | 3.090372 | 3.263940 | 1.866904 | 1.002336 | 5 | 24308.0 |
| eqmotion | matched_selected | 1.586619 | 1.698974 | 0.171734 | -0.381538 | 0 | 8656.7 |

All three guard/control counts match within every recording/frame/seed, not just globally.
Numerically failed matched proposals retain their feasible incumbent and are not called optimal.
Probability zero means no weighted positive event in any visited source leaf, not calibrated impossibility.
Unknown and incomplete futures remain indexed; full-grid bounds are in analysis.json.

## All Registered Contrasts

3000 paired resamples of four physical sites, nominal conditional development intervals.

| Contrast | All gain difference pp, 95% CI | Hard difference pp, 95% CI | Positive-easy difference pp, 95% CI |
|---|---:|---:|---:|
| damped_velocity_005__atom_point_minus_cutoff_point | -0.046326 [-0.079709, -0.017400] | -0.010506 [-0.015317, -0.005444] | -0.082454 [-0.118102, -0.026505] |
| damped_velocity_005__atom_point_minus_matched_point | -0.007144 [-0.012280, -0.002392] | -0.004599 [-0.007369, -0.001828] | 0.029946 [0.010890, 0.049002] |
| damped_velocity_005__atom_population_minus_cutoff_population | -0.203762 [-0.300510, -0.090603] | -0.253881 [-0.431192, -0.090822] | 0.082387 [0.001090, 0.195717] |
| damped_velocity_005__atom_population_minus_matched_population | -0.056698 [-0.081099, -0.029255] | -0.111571 [-0.202211, -0.039954] | 0.159073 [0.037772, 0.273567] |
| damped_velocity_005__atom_selected_minus_cutoff_selected | -0.050859 [-0.089073, -0.017974] | -0.012960 [-0.017605, -0.005514] | -0.085222 [-0.118592, -0.027828] |
| damped_velocity_005__atom_selected_minus_matched_selected | -0.006985 [-0.011283, -0.002688] | -0.004813 [-0.007675, -0.001997] | 0.032019 [0.011411, 0.052627] |
| transformer__atom_point_minus_cutoff_point | -0.072164 [-0.106986, -0.029903] | -0.039153 [-0.082592, -0.003481] | -0.083743 [-0.189482, 0.008757] |
| transformer__atom_point_minus_matched_point | -0.012153 [-0.021400, -0.002907] | -0.010550 [-0.026565, -0.000674] | 0.000162 [-0.026079, 0.015714] |
| transformer__atom_population_minus_cutoff_population | -0.634383 [-0.920428, -0.348339] | -0.628787 [-0.907017, -0.350558] | 0.490990 [0.374270, 0.705613] |
| transformer__atom_population_minus_matched_population | -0.263409 [-0.381698, -0.181993] | -0.329212 [-0.530372, -0.183250] | 0.310475 [0.210116, 0.473533] |
| transformer__atom_selected_minus_cutoff_selected | -0.087892 [-0.136046, -0.034439] | -0.048074 [-0.086638, -0.010085] | -0.077066 [-0.181995, 0.012966] |
| transformer__atom_selected_minus_matched_selected | -0.015791 [-0.028472, -0.003109] | -0.012322 [-0.027888, -0.002733] | 0.000766 [-0.037211, 0.023870] |
| eqmotion__atom_point_minus_cutoff_point | -0.030913 [-0.054644, -0.015029] | -0.003541 [-0.008343, 0.001059] | -0.000165 [-0.013813, 0.014990] |
| eqmotion__atom_point_minus_matched_point | -0.002023 [-0.004503, -0.000224] | -0.000676 [-0.001474, 0.000121] | 0.020799 [0.009418, 0.031889] |
| eqmotion__atom_population_minus_cutoff_population | -0.540568 [-0.946720, -0.265750] | -0.541537 [-1.043171, -0.194695] | 0.500447 [0.188466, 0.955461] |
| eqmotion__atom_population_minus_matched_population | -0.293932 [-0.517551, -0.126054] | -0.355216 [-0.713552, -0.116014] | 0.199511 [-0.109785, 0.770167] |
| eqmotion__atom_selected_minus_cutoff_selected | -0.040391 [-0.077521, -0.017231] | -0.005803 [-0.009185, -0.002420] | 0.006798 [-0.013085, 0.033617] |
| eqmotion__atom_selected_minus_matched_selected | -0.003349 [-0.008677, -0.000449] | -0.000738 [-0.001531, 0.000055] | 0.017156 [0.007643, 0.023875] |

Positive-easy contrasts are differences in gain, so positive means less degradation.
All 54 intervals are nominal, conditional development comparisons, not multiplicity-adjusted claims.
Per-site values and complete machine-readable intervals remain in analysis.json.

## Solver

```json
{
  "instances": 753552,
  "optimal_unverified": 2,
  "incumbent_retained": 0,
  "risk_violations": 0,
  "count_mismatches": 0
}
```
