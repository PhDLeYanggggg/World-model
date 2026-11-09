# WEPDTOF: frozen external forecast replacement test

**Outcome:** The strict joint rule abstained on every clip. At matched
intervention budgets the joint research rule did not demonstrate an
advantage over independent selection. It is not deployable; CV remains
the frozen fallback for this protocol, without counting as a risk pass.

## What was run

- **fresh_run:** 115 disjoint windows from 15 clips and 13 original
  source groups, using 920 real observation images. RAPiD and tracking
  took about 22 minutes including archive I/O. A cold inference of the
  first window reproduced its receipt exactly.
- **fresh_run:** Three donor-only action heads, 800 updates per seed.
  Convenience Store was the only WEPDTOF development source; the
  frozen strict rule made no edits on it. A separate pilot diagnostic
  used fixed counts and had no simultaneous risk and utility pass.
- **cached_verified:** The existing site-excluded OpenDD spline models
  and donor forecasts were hash-checked and reused. No new forecaster
  was fitted on WEPDTOF. Past trajectory scale was computed per row,
  never from held futures.
- **fresh_run:** One held-out label readout after all forecasts were
  frozen. No test tuning. Subsequent aggregate replay reproduced
  the readout byte for byte without rereading labels.
- **not_run:** DroneCrowd independent confirmation (30 sources sealed),
  latent generative rollout, Stage5C execution and SMC.

The [registered protocol](protocol.md) and [fixed-count amendment](readout_amendment.md)
were pushed before outcome access in commit `f1537455f`.
Each query used 8 observed frames and 12 forecast steps sampled at
stride four. Units are image pixels; no metric, 3D or seconds-level
claim. Both direct controls and proposals use the same online roster.

## What the comparison says

| Arm | Known-step ADE (px) | Available-endpoint FDE (px) | Lower gain vs CV (pp) | Lower gain vs OLS4 (pp) | Original risk passes | Risk and both utility passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CV / strict joint / guarded | 43.90 | 83.91 | 0.00 | -66.16 | 0/45 | 0/45 |
| OLS4 | 37.92 | 71.69 | -46.82 | 0.00 | 3/45 | 0/45 |
| Direct spline | 45.30 | 84.01 | -44.85 | -40.54 | 0/45 | 0/45 |
| Fixed-count joint | 43.65 | 83.35 | +0.44 | -65.38 | 4/45 | 0/45 |
| Fixed-count independent | 43.64 | 83.35 | +0.45 | -65.36 | 4/45 | 0/45 |
| Fixed-count common family | 43.66 | 83.37 | +0.40 | -65.39 | 3/45 | 0/45 |
| Fixed-count no pair penalty | 43.65 | 83.35 | +0.44 | -65.38 | 4/45 | 0/45 |

The conservative full-grid scores use common CV-reference mass, in
percentage points, and debit unseen future positions. They are not
relative percent changes in visible ADE. Opposite comparisons can
both have negative lower bounds. OLS4 is more accurate on the visible
steps; the stricter statement that it is a safe replacement is not
supported. The four fixed-count arms each make
366 edits total with the same per-query counts and peak displacement.

| Direct paired contrast, fixed-count joint minus | Conservative pp | 95% source bootstrap | Estimable original sources |
| --- | ---: | --- | ---: |
| Independent | -0.045 | [-0.113, +0.012] | 10/13 |
| Common family | -0.021 | [-0.106, +0.051] | 10/13 |
| No pair penalty | 0.000 | [0.000, 0.000] | 10/13 |

The direct contrasts use pair-specific missing-label debits, so do not
subtract two independently computed lower bounds to reconstruct them.
Intervals use 3,000 resamples of original sources after averaging clips
and seeds. Empty strict contrasts are not estimable, not zero effects.
The no-penalty result is an empirical tie under this fixed setup, not
evidence of general equivalence.

## Failure diagnosis

There were 431 online track-windows across 92/115 windows. Of 1,005
annotated history instances, 353 had stable camera matches. Of the
431 online histories, 347 had full futures, six had partial futures,
and 78 were wholly unobserved; 57 of the last group came from the
exhibition clip. No match is not proof of a false detection.
All online histories stay in the original risk audit.

Fixed-count joint had 9 views without edits, 12 edited views with
undefined easy reference, 20 with observed risk failure and 4 risk
passes; none of the four also beat both baselines. These categories
are exclusive in the breakdown; a view lacking easy support may
also exceed another measured risk bound. The worst estimable signed
easy degradation was 7.44%. Only 10/45 views had positive
conservative utility against OLS4; 34 fixed edits across seeds
affected histories with no observed future. The median missing-label
debit versus CV was zero, so coverage is not the sole explanation.
The strict rule's complete abstention and the fixed comparison's
weak utility are separate failures.

For a next development experiment, compare a camera-matched,
causal reference forecast against OLS4 before training a more
complex joint ranker. Check whether the action bank can improve both
baselines at the same displacement, and whether online track quality
identifies usable histories. Use development sources only; these
13 groups are now exposed and cannot be reused as a new confirmation.

[Aggregate readout](external_readout.json) |
[Failure decomposition](failure_breakdown.json) |
[Camera freeze](../wepdtof_external_camera_v1/camera_freeze.json) |
[Chinese reproduction](../../../reproducibility/M3W_CURRENT_ZH.md).
