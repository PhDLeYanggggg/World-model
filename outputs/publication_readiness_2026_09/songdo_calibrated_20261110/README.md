# Songdo: frozen source-disjoint confirmation

I froze the [40/160 protocol](protocol.md) before opening any outcomes
(`ebf943c6a`) and published the [calibration freeze](calibration_receipt.json)
before confirmation (`0c7919432`). [Songdo Traffic v2](https://zenodo.org/records/17924857)
provides the release trajectories described by [Fonod et al.](https://www.sciencedirect.com/science/article/pii/S0968090X25002098).
The experiment uses the first acquisition day, 40 calibration flights
in three linked components and 160 separate confirmation flights in
fourteen. All 480 confirmation flight-seed views have matched proposals.
The linked components, not flights, overlapping trajectories or seeds,
are the resampling units. This is one acquisition day in one city.

## Matched proposals: same query, two edits, same radius

| Joint minus | Observed contribution (pp) | Conservative gain (pp) | 95% component bootstrap (pp) |
| --- | ---: | ---: | ---: |
| Independent | -0.000272 | -0.000442 | [-0.001499, +0.000237] |
| Additive | -0.000278 | -0.000448 | [-0.001476, +0.000193] |
| Same family | +0.000217 | +0.000141 | [+0.000021, +0.000306] |

There is a small positive contrast against the same-family restriction,
not against either unrestricted control. Relative-motion lower bounds
have intervals spanning zero against all three controls. For joint minus
independent the point is -0.000282 native units, with an interval of
[-0.000701, +0.000129]. These are source-level intervals from 3,000 draws
after averaging seeds within flight. The gain points use the original
observed CV denominator; native coordinate errors are not pooled with
pixel errors from other benchmarks.

| Matched proposal | Available ADE | Available FDE | Original risk passes / 480 | Risk and both utility passes / 480 |
| --- | ---: | ---: | ---: | ---: |
| CV, no-edit reference | 1.373135 | 3.416751 | Not an edited-risk test | Not an edited-risk test |
| OLS4, accuracy reference | 1.603831 | 3.808624 | Not evaluated for release | Not evaluated for release |
| Independent | 1.373111 | 3.416737 | 179 | 169 |
| Additive | 1.373110 | 3.416737 | 173 | 162 |
| Same family | 1.373127 | 3.416776 | 71 | 56 |
| Joint | 1.373119 | 3.416759 | 69 | 53 |

These counts are retrospective properties of fixed-count proposals,
not release successes. Joint has 336 undefined risk views and 177 with
measured risk violations; the counts can overlap. In this cohort CV
is more accurate than OLS4. The approximate +10.44 pp against OLS4
in released outputs is already present in the CV fallback.

## Actual frozen release

Calibration uses 71,105 complete positive-CV-error examples, with easy
and hard cutoffs 0.087787 and 2.294398 in native units. Joint, additive
and same family have no release threshold for any seed; each returns
CV in all 480 confirmation views. The independent seed-29 threshold
releases 15 views: five pass original risk, four also beat both
references, ten have undefined risk and one has a measured violation
(the latter two categories can overlap). Its whole-roster conservative
gain over CV is only +0.0000163 pp. No arm qualifies for deployment.

For the CV fallback, the source-averaged flight p95 ADE is 6.66159,
the worst source mean ADE is 1.70887 and the worst single flight-seed
ADE is 2.92736, all in native units. Edited easy or hard summaries
without full component support remain undefined. The [aggregate
receipt](confirmation_receipt.json) includes the full metrics, masks'
support counts, and paired component means without flight identifiers.

```sh
env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 .venv-pytorch/bin/python scripts/export_m3w_songdo_confirmation.py
.venv-pytorch/bin/python -m pytest -q tests/test_m3w_songdo_confirmation_export.py tests/test_m3w_songdo_confirmation.py tests/test_m3w_songdo_matched_readout.py tests/test_m3w_songdo_secondary.py tests/test_m3w_songdo_future_join.py
```

The exporter reproduces the primary report from the 160 local flight
receipts and verifies every release against the committed pre-outcome
freeze. It does not open labels. Twenty-two tests passed. The Songdo
roles and scores are frozen and cannot be reused to tune a new model
and still call this confirmation. DroneCrowd confirmation stays sealed.
