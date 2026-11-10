# Why the frozen Songdo joint rule did not release

The release rule chooses a high-score prefix in which every calibration
proposal passes all three original 2% checks. Of the 40 calibration flights,
31 have matched proposals at each seed. In the joint arm, 7, 5 and 3 pass
risk for seeds 17, 29 and 43, respectively. The highest-scoring proposal
in each seed fails only because its selected-easy risk denominator is
undefined. Thus the admissible prefix is empty even though there are
passing proposals further down the same list. Across all four arms and
three seeds, 11 of 12 first blocking score groups have only this missing
denominator; one has a measured violation. The independent arm at seed 29
alone has a nonempty prefix.

On the unchanged 160 confirmation flights, I compared top-ranked and
uniform proposals at the same count within each source component and seed.
The ranked diagnostic selects 102 of 480 views per arm. Ties share the
last slots uniformly. The rate averages seeds within each of the 14 source
components, then gives each component equal weight. Uniform is its exact
same-budget expectation, not a newly sampled policy.

| Proposal rule | Ranked dual-pass rate | Uniform expectation | Difference (pp; 95% paired source interval) |
|---|---:|---:|---:|
| Independent | 38.33% | 35.95% | +2.38 [-8.10, +13.33] |
| Additive | 37.14% | 34.40% | +2.74 [-6.90, +12.74] |
| Shared family | 13.33% | 11.79% | +1.55 [-4.05, +6.90] |
| Joint | 12.14% | 11.19% | +0.95 [-3.10, +4.64] |

Dual pass means passing the original risk checks **and** improving on CV
and OLS4 under the conservative score. The joint ranker reduces measured
violation by 8.33 pp (ranked minus uniform [-13.93, -2.74]) but increases
undefined risk by 9.05 pp [+2.86, +15.00]. These outcomes can overlap.
Fewer measured violations do not show safer forecasts if risk becomes
harder to estimate. The primary dual-pass interval includes zero; the
intervals for these additional diagnostics are not multiplicity adjusted.

This locates the gap in risk support and ranking, not an empty menu. A
future rule can predict reference availability separately and compare
against the same score on a new, independent cohort; this readout does not
establish that such a rule would work. In particular, skipping the failed
prefix items after seeing confirmation would change the study. The original
joint release count remains zero, and no release rule was changed.

```sh
python3 -I -B -S scripts/audit_m3w_songdo_score_ranking.py
```

Fresh analysis of hash-verified anonymous score summaries. No trajectory
labels, fitting, rethresholding or additional confirmation. [Protocol](protocol.md)
and [full readout](readout.json). The original studies of
[release](../songdo_calibrated_20261110/README.md) and
[hindsight capacity](../songdo_frozen_menu_capacity_v1/README.md) remain unchanged.
