# Does Better Label Coverage Rescue Forecast Selection?

**No deployable result.** The learned selector beats a matched hindcast
heuristic but does not beat last-displacement CV. Nearest-neighbor history
hurts at the same intervention count. This is a fresh OpenDD location-held
diagnostic, not independent confirmation of a frozen DroneCrowd policy.

## Actual Run

- Seven leave-one-location-out folds, three seeds and two matched models:
  42 fresh Torch fits. Each fold trains only on the other six sites; all
  1,819 source pedestrian IDs were disjoint across flights and sites.
- The past-only schedule found 3,395 non-overlapping query intervals from
  202 flights, with 10,475 pedestrian targets. Complete 12-step labels cover
  9,823 targets (93.78%); 599 were partial and 53 had no future labels.
- Each matched arm edits one trajectory in each of the 3,374 queries with
  any legal nonidentity option. Neither future availability nor the future
  positions determine those queries or edit counts. The oracle alone uses
  outcomes to choose the recipient and candidate.
- Local arm64 Torch used four compute threads, one interop thread, and no
  workers. A fresh run completed in 20.45 seconds after entry into the runner.
  The checkpoint readout replay matched the result byte for byte. Replay
  did not refit the models or provide another independent experiment.

## Results

Equal weight per site, after averaging seeds. ADE/FDE use observed labels in
publisher UTM coordinate units. Gain is the missing-step conservative
paired difference divided by observed OLS4 error mass, in relative percentage
points. The intervals resample the seven sites 3,000 times. They describe
variation across these sites, not independent-city or publication-ready
confirmation. CV and OLS4 are no-edit references; the other arms share the
same motion menu and switch count.

| Method | Observed ADE | Observed FDE | Gain lower bound vs OLS4, pp [95% site bootstrap] |
|---|---:|---:|---:|
| Last-displacement CV | 0.6083 | 1.3534 | +6.211 [+3.441, +8.325] |
| OLS4 | 0.6619 | 1.4332 | 0 |
| Minimum change | 0.6586 | 1.4276 | +0.385 [-0.119, +0.844] |
| History hindcast | 0.6760 | 1.4748 | -2.943 [-7.718, +1.313] |
| Learned self-only | 0.6415 | 1.4020 | +2.312 [-0.047, +4.350] |
| Learned nearest-neighbor | 0.6432 | 1.4060 | +2.071 [-0.104, +3.832] |
| Matched observed-label oracle (not deployable) | 0.5314 | 1.1742 | +19.074 [+17.321, +20.947] |

At the exact same counts, self-only beats hindcast by **+4.334 pp**
[+1.795, +7.154]. Adding nearest-neighbor features costs **-0.480 pp**
[-0.773, -0.154] relative to self-only, using the two actual forecasts to
debit missing steps. The best learned arm loses **-4.681 pp** to CV on
observed steps with the same OLS4 denominator. Even an optimistic bound
on this finite roster is **-2.577 pp**: take the learned arm's paired
upper bound against OLS4 and subtract CV's paired lower bound. This bound
is not a confidence interval. On complete rows CV also has lower ADE
(0.6168 vs. 0.6509 self-only).

| Site | Rows | Flights | Self-only lower gain vs OLS4, pp | Self-only signed easy ADE change vs CV |
|---|---:|---:|---:|---:|
| rdb1 | 5,349 | 127 | +2.679 | +21.47% |
| rdb2 | 50 | 2 | +1.865 | +22.55% |
| rdb3 | 82 | 5 | -3.760 | +11.82% |
| rdb4 | 113 | 7 | +1.483 | +17.78% |
| rdb5 | 322 | 15 | +3.701 | +35.53% |
| rdb6 | 4,547 | 45 | +3.518 | +38.64% |
| rdb7 | 12 | 1 | +6.697 | -15.37% |

The descriptive easy slice worsens by more than 2% on six sites for both
learned arms. The apparent improvement at rdb7 is based on two easy targets
in one flight. This is not the original per-view deployment test and does
not replace or relax it. Two sites supply 94.47% of the diagnostic rows;
site weighting prevents dominance but does not create more independent
locations. Raw-video tracking causality was not verified. Neither this
diagnostic nor its oracle establishes a multimodal or joint-selection gain.

## Research Decision

Keep the deployed policy unchanged and DroneCrowd confirmation sealed.
Better label support did not make this OLS4-centered policy beat the
stronger reference. The next defensible hypothesis is to learn incremental
gain against CV and OLS4 directly, with baseline choice and the decision to
abstain fixed on training data. OpenDD has now been read and may be used
for method development, but a model changed in response to this result
needs independent confirmation before any new generalization claim.

```sh
.venv-pytorch/bin/python scripts/run_m3w_opendd_location.py --register-only
.venv-pytorch/bin/python scripts/run_m3w_opendd_location.py
```

Rerunning resumes hash-bound site caches and completed optimizer checkpoints
under `data/stage_cvpr2027_experiments/opendd_location_diagnostic_v1/` and
requires an exact match to existing analysis. The heartbeat records PID,
site and checkpoint step. The runner takes a single-process lock; check the
recorded PID before resuming. Archives, trajectories, caches and weights are
not distributed. The [publisher](https://l3pilot.eu/data/opendd.html) documents
the coordinates and license. SDD remains pixel/raw-frame only. Stage5C and
SMC remain disabled.

[Frozen protocol](protocol.md) | [Registration](registration.json) | [Complete results](analysis.json)
