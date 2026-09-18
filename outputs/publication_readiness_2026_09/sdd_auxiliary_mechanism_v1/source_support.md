# Source Event Support and Permutation Checks

Training-source diagnostic only, zero optimizer updates; held main labels are not evaluated.
Exact-static/stop/turn proxies are not human intent labels. Partial support is not silently discarded.

| Complete-label event | Windows | Recording-local tracks | Recordings |
| --- | ---: | ---: | ---: |
| static_stays | 12335 | 612 | 36 |
| static_moves | 10039 | 342 | 32 |
| moving_stops | 20257 | 811 | 39 |
| moving_turns | 23593 | 1509 | 39 |
| other_motion | 122134 | 2677 | 40 |

Partial/absent-label windows: 40,975; retained in training, unclassified in this complete-label event diagnostic.
The stricter historical half-box-displacement start proxy has 244 train-only windows from 58 recording-local tracks.
The exact-nonzero training category and half-box census proxy are different labels; counts are not interchangeable.
Scale-floor and target-magnitude diagnostics compare admitted SDD with main fold-0 training rows only; see JSON.
A fixed 0.001 native-coordinate floor is not a verified pixel-to-local-unit calibration. No normalization change is made.
Counts of actual source draws/unique event rows per seed are in source_support.json.
Permutations preserve all 229,333 row identities as a bijection within recording and target-support strata.
For each seed, 128 real source rows have unchanged inputs and exact predictions in all three modalities.
Donor singleton/same-track counts remain in input_checks.json; no complete independence claim.
Source event support is measured, not used to alter this registered training run.
