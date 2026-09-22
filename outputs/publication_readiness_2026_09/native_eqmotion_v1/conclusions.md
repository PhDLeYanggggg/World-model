# Matched Native-Loss EqMotion: Better Accuracy, Failed Protection

Readout completed 2026-09-22 Europe/London. The registration was committed as
`33ecf02b` before the real pilot and full training. No endpoint was selected by
its held-site score. All twelve fits completed before the first new readout.

## Evidence Status

- `fresh_run`: twelve native-loss EqMotion K=1 fits, 48,000 updates, 3,072,000
  draws and 527,268 held-source predictions. Recorded fitting time totals
  17,715.45 seconds (4.92 hours), including the resumed pilot.
- `cached_verified`: native Transformer checkpoints/predictions, admitted
  causal inputs, fitting identities and row-level sampling counts.
- `not_run`: a new EqMotion intervention head, independent calibration or
  confirmation, deployment, metric/time calibration, Stage5C and SMC.
- Complete replay of all twelve checkpoints and 527,268 predictions passes.
  The separate arithmetic implementation verifies twelve matched fits,
  1,581,804 repeated fitting rows and 320 scene reductions, including the paired
  interval. This is a second implementation by the same agent, not independent
  research confirmation. All 35 related tests pass. Required processes exited0.

The task observes eight annotation steps and predicts twelve, SDD stride12.
This is not the historical raw-frame t+50 task. The 175,756 past-eligible target
queries span 33 recordings and four design-exposed physical sites. Excluding a
site from its model fit does not erase its prior role in research decisions.
Results are pixel-coordinate development evidence, not seconds, verified metric
prediction, true 3D, foundation-model success or independent generalization.

## Main Result

Gains are percentages relative to causal constant velocity (CV), averaged equally
over the four sites. Positive-easy degradation is an error increase; lower is
better. Seed-mean results average errors, not predicted coordinates, and are not
a trained ensemble. The confidence intervals resample four physical sites 3,000
times after seed averaging. They are conditional development intervals, not
independent confirmation or 175,756 independent observations.

| Fixed predictor | ADE gain | FDE gain | Hard ADE gain | Positive-easy degradation | Complete zero-CV harms per seed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Causal constant velocity | 0.00000% | 0.00000% | 0.00000% | 0.00000% | 0 |
| Constant position | -213.08350% | -182.60131% | -85.86572% | 836.19793% | 7 (deterministic control) |
| Frozen native Transformer | 7.63308% | 8.64511% | 10.65751% | 21.70968% | 3,000 |
| Fresh native EqMotion K=1 | 11.04350% | 12.39109% | 15.40148% | 35.24968% | 11,528 |

EqMotion's ADE-gain interval is [8.22457%, 13.39760%]. The registered primary
EqMotion-minus-Transformer contrast is **+3.41042 percentage points**, paired
site interval **[+0.93607, +6.06916]**. The local Transformer does not win this
matched accuracy comparison. Both neural systems fail the easy-preservation
criterion and exact-zero-CV protection. Neither is newly deployed.

| Seed | EqMotion ADE gain | Transformer ADE gain | EqMotion positive-easy degradation | Transformer positive-easy degradation |
| --- | ---: | ---: | ---: | ---: |
| 17 | 11.29256% | 7.72587% | 34.57514% | 21.27426% |
| 29 | 10.64997% | 7.87493% | 37.55001% | 21.04803% |
| 43 | 11.18797% | 7.29844% | 33.62388% | 22.80673% |

| Site | EqMotion ADE gain | EqMotion minus Transformer gain |
| --- | ---: | ---: |
| coupa | 10.58439% | +1.93718 pp |
| deathCircle | 12.51506% | +7.44649 pp |
| gates | 6.79440% | -0.06504 pp |
| hyang | 14.28015% | +4.32304 pp |

All seed-level aggregate gains are positive against CV; EqMotion wins three of
four site-mean comparisons with Transformer, not every site. Raw per-site
ADE/FDE, tails, coverage and every seed remain in [analysis.json](analysis.json).

## Failure Taxonomy

1. **Average prediction is not safe intervention.** EqMotion improves hard
   ADE by 15.40%, but degrades positive-easy ADE by 35.25%. Better average dynamics
   therefore does not resolve the project's central harm-control problem.
2. **Stationary-history drift remains material.** EqMotion's static-history
   gain is -50.79774%, while this Transformer control is exactly CV on that
   subset. The fixed systems differ in their output wrapper, so this is not an
   isolated test of equivariance or proof that equivariance causes drift.
3. **Exact-reference protection fails.** There are 11,566 complete outcomes
   with exactly zero CV ADE. EqMotion harms 11,528 per seed (34,584 repeated
   query/seed instances); Transformer harms 3,000 per seed (9,000 instances).
   Seed-mean EqMotion absolute ADE on those outcomes is 1.01961, 3.64118,
   2.37003 and 0.83461 annotation pixels in the four sites, respectively.
   Percentage harm is undefined at zero reference error. No epsilon is added.
4. **Outcome support is incomplete.** Of 175,756 queried rows, 172,957 support
   ADE and 144,010 support the requested endpoint FDE; 143,918 have a complete
   future grid. The 2,799 absent ADE and 31,746 absent endpoint labels are not
   zero-error successes. Full-population protection is not established.
5. **Comparator evidence is not method novelty.** The public EqMotion core
   is stronger on average in this fixed setting. The M3W paper cannot present
   the local Transformer as a superior prediction architecture on these data.
   Nor does observing a tradeoff establish a new solution to it.

Hard uses the fitting-only CV q75 and positive-easy the fitting-only positive-CV
q25, applied to evaluation errors. These are diagnostic outcome subsets, never
inference inputs. Complete zero-CV protection retains the exact original rule.

## What Is Matched and What Is Not

All twelve pairs use identical fitting IDs, row-level draw counts, native-loss
factors, 4,000 updates and batch64. No held-site row is sampled for fitting.
The author core is pinned at `5aec2e0b61c511fa93a24138dd90da59a089084b`.
Head0 was fixed before training; there is no future-informed best-of-K selection.

This is a fixed-system comparison, not parameter/FLOP matching or a pure
equivariance ablation. EqMotion has 580,676 trainable parameters and 3,027,268
stored parameters including unused frozen heads. The existing Transformer has
its motion-bound wrapper; no new wrapper was added to EqMotion. This study is
not reproduction of the paper's published minADE20/minFDE20 benchmark.

## Decision and Next Evidence

Keep the matched EqMotion result as a required strong comparator and retain the
Transformer loss-repair result as a controlled within-family finding. Do not
promote either uncontrolled neural forecast. Do not retrospectively retune
thresholds or claim a test win from these explored sites.

The useful next experiment is a preregistered, lineage-clean comparison of the
same intervention mechanism across these fixed predictor families, including
simple past-stop and matched-intervention controls. EqMotion cost supervision
would require the same pair-excluded producer discipline as Transformer; the
new held-source outcomes must not be recycled as an independent calibration set.
Whether bounded cost learning transfers to the stronger predictor is **not yet
tested**. Independent source-use/exposure review and approved calibration/final
roles remain necessary before any confirmation claim.

Submission readiness remains unmet: a distinct reliable-intervention contribution,
independent evidence, and final paper/reproducibility packaging are still needed.
No data roles, deployment policy, risk tolerance, Stage5C or SMC are changed.
