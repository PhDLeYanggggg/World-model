# Choosing Who to Edit and When

I kept the frozen OpenDD EqMotion trajectories, the held-site release
stream, the two-query expiry, and the original recording risk rule.
At each legal edit count, I offered up to five distinct recipient
sets from causal ranks: CV/OLS disagreement, large and small neural
trajectory change, observed turning, and nearest observed neighbor.
The fixed disagreement prefix is always included. No future
positions or recording end enter the inference features.

I trained a small network to score both immediate and continuation
value. Candidate trajectories and release heads for the fitting
rows exclude both the donor site and the outer evaluation site.
Actions of the same size share the queue transition; their targets
differ by the slate's immediate return. Training retains the
previous additive lower-gain and risk-cost proxy. That proxy is not
the original hard recording constraint.

The full head sees predicted relative separation summaries for
edited-edited and edited-unedited agent pairs. The no-geometry head
is refit without those ten features. Both still have causal motion
features and the same menu, including the observed-neighbor rank.
This tests incremental value of relative rollout geometry, not
the absence of all interaction information.

## Causal Results

| Policy | Edits | Undefined views | Risk passes | Joint passes |
| --- | ---: | ---: | ---: | ---: |
| Fixed rank, immediate | 3,323 | 438 | 11 | 5 |
| Adaptive slate, immediate | 3,323 | 403 | 13 | 10 |
| Fixed rank with learned queue | 1,829 | 443 | 25 | 14 |
| Myopic adaptive queue | 2,458 | 407 | 19 | 13 |
| Joint recipient and time | 2,280 | 401 | 26 | 16 |
| Retrained without relative geometry | 2,190 | 410 | 23 | 15 |
| Fixed rank, identical realized query counts | 2,280 | 409 | 29 | 16 |
| Adaptive immediate, same realized credits | 2,280 | 438 | 12 | 9 |

Joint pass means the original 2% risk and support rules pass, and
both conservative gains versus CV and OLS4 are positive. Neither
retrospective matched comparison certifies the full policy.

| Contrast | Lower gain, pp [95% site-bootstrap interval] |
| --- | ---: |
| Joint vs fixed rank at identical query counts (primary) | -0.008 [-0.085, +0.061] |
| Joint vs adaptive immediate at the same used recording totals | -0.042 [-0.279, +0.193] |
| Joint vs myopic adaptive queue | -0.172 [-0.379, -0.041] |
| Joint vs retrained without relative rollout geometry | -0.136 [-0.384, +0.029] |
| Joint vs CV | -0.190 [-0.622, +0.164] |
| Adaptive immediate vs fixed immediate | -0.145 [-0.663, +0.224] |

The fixed-clock comparison substitutes 1,048 recipients, so its
near-zero gain is not due to identical predictions. The corresponding
lower gain on relative positions is +0.054 pp [-0.240, +0.403].
The joint model's worst site versus CV is -1.226 pp.
Gain is normalized by observed CV error mass; relative-position gain
uses the fixed-clock relative error mass. The 3,000 bootstrap draws
resample seven sites after averaging three seed views. The 606
views are 202 exposed recordings, not independent confirmation.

## Where the Gain Is Still Lost

| Outcome-aware ceiling | No delay | Up to two queries |
| --- | ---: | ---: |
| One fixed rank, all originally released credits | 5 | 84 |
| Five-rank menu, all originally released credits | 65 | 211 |
| Any legal recipient, all originally released credits | 88 | 223 |

The causal menu recovers 127 of the 139 views lost by imposing one
rank at two queries. This is a property of the available choices, not
the learned rule. Under the learned rule's *actual* clock and its
2,280 realized edits, the exact outcome-aware menu yields **107/606**
joint passes against **16/606** for the learned rule. The original
scorer verifies every positive witness; no solve is unresolved.
The 107 is a separate matched-clock limit, not a subset decomposition
of the 211 at full released totals. There are 244 no-edit views and
68 with no reachable easy support under the realized clock.

The ranking gap remains even after holding timing and counts fixed.
I would prioritize within-count pairwise training on actual
relative gain and separate harm targets before another change to
the queue. The current regression of an additive proxy did not
learn a ranking that uses the feasible space. Neither broader
options nor relative geometry alone is evidence of a better policy.

## Reproduce

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_slate_queue.py
.venv-pytorch/bin/python scripts/run_m3w_opendd_slate_queue.py --replay
.venv-pytorch/bin/python scripts/audit_m3w_opendd_slate_clock.py
.venv-pytorch/bin/python scripts/build_m3w_opendd_timing_replay.py
```

`fresh_run`: 42 local arm64 Torch fits, 21,000 optimizer updates,
31.6 seconds, and both exact menu analyses. Four threads and zero
data workers; a checkpoint and heartbeat every 100 steps. The
original command resumes. `cached_verified`: the candidate bank
and release heads; the 21.6-second zero-update replay reproduces
the experiment. The anonymous scalar archive verifies 27 count
tables and 21 intervals across three timing studies; it is not
full training or upstream forecast reproduction.

No CREATE job was needed. The 15/30 DroneCrowd roles were verified
without reading confirmation labels. No deployment, Stage5C or SMC.
Native 8/12 annotation steps, dataset-local coordinates, no metric
geometry or seconds claim.

[Registration](registration.json) | [Site and seed results](analysis.json) |
[Clock registration](clock_registration.json) | [Exact clock results](clock_capacity.json)
