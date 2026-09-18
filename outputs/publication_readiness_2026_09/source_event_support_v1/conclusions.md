# Current-Cohort Annotation Event Support

## Conclusion

The current stationary-history training subset contains many overlapping windows
but substantially fewer annotation episodes. Missing neighbor slots are not the
main supported explanation for the large-excursion failure. Reweighting existing
forecasts by episode or track does not create a positive result.

This is a fresh raw-annotation audit over hash-verified earlier context and
predictions, not a rerun of the full-SDD census and not a new trained model.

## Support Counts

| Diagnostic subset | Windows | Scoped tracks | Past-defined episode groups | Disjoint within-track full-window spans |
| --- | ---: | ---: | ---: | ---: |
| All current queries | 15,430 | 545 | 1,457 | 1,443 |
| No sampled future change | 8,566 | 453 | 643 | 860 |
| Any sampled future change | 6,864 | 253 | 1,129 | 848 |
| Excursion above 10 annotation pixels | 728 | 115 | 151 | 142 |
| Excursion at least half past-box diagonal | 207 | 47 | 55 | 52 |
| Outside half-box radius for final four steps | 113 | 33 | 38 | 35 |

Subsets overlap and must not be added. Episode groups are maximal past runs of
constant annotation centers within a recording-scoped track; gaps/lost states
break a run. They are not verified physical events or independent people.
Three hundred fifteen groups contain both positive and negative query labels:
the same static interval can be far from, or close to, a future sampled change.
The mean group supplies 10.59 windows; the median is seven and the maximum 56.

Half-box excursion support by site is 12/19/3/21 groups for
coupa/deathCircle/gates/hyang, from 10/18/2/17 scoped tracks. Gates has only two
relevant track IDs. These limits cannot be repaired by multiplying bootstrap
draws or treating recording-local IDs as independent people across videos.

## What Was Actually Observed

- All 207 half-box windows have a current neighbor and a complete eight-step neighbor slot; 206 also have a moving neighbor. Availability does not prove that its motion predicts the focal agent.
- The half-box subset has 99.879% mean crop coverage, but the median annotation-box footprint is only 7.10 by 11.27 output pixels. Its mean temporal embedding-energy fraction is 3.771%. Image change is not verified intent.
- 14,979 histories are static on every raw annotation frame in the observed interval. The other 451 only appear static on the eight sampled steps; they remain in the cohort.
- All 15,430 future intervals have 144 non-lost raw-frame labels. There are 7,006 raw-change windows versus 6,864 sampled-change windows: 142 return to the same sampled centers between prediction steps. This does not authorize changing the target grid.
- The median count of ungenerated controls anywhere in the past interval is zero, both overall and for half-box excursions. Supplied interpolation and the earlier later-control warning remain explicit; this is not strict sensor-as-of perception or human-gold labeling.

## Reweighted Frozen Forecasts

These are post-hoc descriptive weightings of already fixed predictions, not a
replacement main metric. All rows and the original equal-site score are retained.

| Frozen arm | Original window-based equal-site gain | Equal episode within site | Equal track within site |
| --- | ---: | ---: | ---: |
| Geometry | -0.0704% | -0.0213% | -0.0364% |
| Current appearance | -1.9079% | -0.6589% | -1.1802% |
| Sequence appearance | -6.1022% | -2.2023% | -3.6198% |
| Centered sequence | -0.7622% | -0.2480% | -0.4610% |
| Centered + RMS | -1.7564% | -0.5888% | -1.0379% |

No positive arm emerges. Contributions are concentrated differently under the
three descriptions, but neither episode counting nor retrospective weighting
establishes more predictive information. The four source sites remain explored.

## Verification And Next Experiment

All 29 raw recordings and all cached row identities are checked. Past features
are unchanged under future corruption and truncation for 1,077 real queries.
The full audit was rerun and reproduced the immutable row archive and aggregate
JSON exactly. Six new helper tests plus existing quality/image tests pass
(16 combined). No new forecast, fitting, main-role score or deployment occurs.

A separate fixed experiment tests equal episode exposure during training, with
geometry and centered models, all original rows and the original evaluation.
It changes only the training sampler and cannot create new independent events.
See [registered design](../source_episode_sampler_decision.md). Its results must
be reported separately, including failure; this audit does not predict success.

Reproduce with `.venv-pytorch/bin/python scripts/audit_m3w_source_event_support.py`.
The script requires the private hash-matching row/image audits and fixed forecast
archives. [audit.json](audit.json) contains all sites, seeds and evidence hashes.
Private event rows are not published. Annotation-pixel/raw-frame only, no metric,
seconds, true-3D or foundation claim. Stage5C and SMC remain off.
